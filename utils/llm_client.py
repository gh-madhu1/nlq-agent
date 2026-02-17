import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline, GenerationConfig

load_dotenv()
logger = logging.getLogger(__name__)

# Module-level singleton cache: load the model ONCE, reuse forever
_llm_cache: dict[str, object] = {}


def _detect_device():
    """Auto-detect the best available device: MPS > CUDA > CPU."""
    import torch
    if torch.backends.mps.is_available():
        logger.info("Device detected: MPS (Apple Silicon GPU)")
        return "mps", torch.float16
    elif torch.cuda.is_available():
        logger.info("Device detected: CUDA GPU")
        return "cuda", torch.float16
    else:
        logger.info("Device detected: CPU")
        return "cpu", torch.float32


class LLMClient:
    def __init__(self, model_provider: str = "openai", model_name: Optional[str] = None):
        self.provider = model_provider.lower()
        self.model_name = model_name
        self.tokenizer = None
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Load or retrieve the cached LLM. Model weights are loaded exactly once."""
        cache_key = f"{self.provider}:{self.model_name}"

        if cache_key in _llm_cache:
            logger.info(f"Reusing cached LLM: {cache_key}")
            return _llm_cache[cache_key]

        logger.info(f"Loading LLM for the first time: {cache_key}")
        llm = self._build_llm()
        _llm_cache[cache_key] = llm
        return llm

    def _build_llm(self):
        """Build the LLM pipeline — called only once per (provider, model) pair."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment. Using dummy key for initialization.")
                api_key = "sk-dummy"
            return ChatOpenAI(
                model=self.model_name or "gpt-4o",
                openai_api_key=api_key,
                temperature=0
            )
        elif self.provider == "vllm":
            # vLLM is best used via its OpenAI-compatible server
            api_key = os.getenv("VLLM_API_KEY", "EMPTY")
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            
            logger.info(f"Connecting to vLLM server at {base_url}")
            return ChatOpenAI(
                model=self.model_name or "meta-llama/Llama-3.2-3B-Instruct",
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0
            )
        elif self.provider == "local":
            model_id = self.model_name or "meta-llama/Llama-3.2-3B-Instruct"
            logger.info(f"Downloading/loading local model: {model_id}")

            device, dtype = _detect_device()
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "use_cache": True,        # KV caching for faster autoregressive decoding
            }

            # Device-specific configuration
            if device == "cuda":
                # Use 4-bit quantization on CUDA for memory efficiency
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif device == "mps":
                # Native float16 on Apple Silicon — no quantization needed
                model_kwargs["torch_dtype"] = dtype
                model_kwargs["device_map"] = "mps"
            else:
                # CPU fallback — use float32 for compatibility
                model_kwargs["torch_dtype"] = dtype

            # Use SDPA attention if available for speed
            try:
                model_kwargs["attn_implementation"] = "sdpa"
            except Exception:
                pass

            # Manual loading to control generation_config and silence warnings
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )

            # Explicitly create generation config for greedy decoding
            generation_config = GenerationConfig.from_pretrained(model_id)
            generation_config.max_new_tokens = 512
            generation_config.do_sample = False
            generation_config.repetition_penalty = 1.2
            generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id
            
            # This is the key: set the model's generation_config to our explicit one
            model.generation_config = generation_config

            pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.2,
                return_full_text=False,
            )
            
            self.tokenizer = tokenizer
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")

    def get_llm(self):
        return self.llm

    def call(self, prompt: str, system_prompt: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 512, 
             stop: list = None, max_retries: int = 2) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            max_retries: Number of retries on OverflowError
        """
        start_time = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "openai":
                    if system_prompt:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=prompt)
                        ]
                        response = self.llm.invoke(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                    else:
                        response = self.llm.invoke(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                    result = response.content if hasattr(response, 'content') else str(response)
                elif self.provider == "vllm":
                    if system_prompt:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=prompt)
                        ]
                        response = self.llm.invoke(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                    else:
                        response = self.llm.invoke(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                    result = response.content if hasattr(response, 'content') else str(response)
                else:
                    # Local model (HuggingFace)
                    # Use chat template if tokenizer is available
                    if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        
                        templated_prompt = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    else:
                        templated_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                    # Constrain max_tokens to prevent overflow
                    safe_max_tokens = min(max_tokens, 256)  # Limit to 256 tokens for safety
                    
                    result = self.llm.invoke(
                        templated_prompt,
                        max_new_tokens=safe_max_tokens,
                        stop_sequences=stop or []
                    )
                
                # Strip echoed prompt if present (some models/configurations echo the prompt)
                result_str = result.content if hasattr(result, 'content') else str(result)
                if result_str.startswith(prompt):
                    result_str = result_str[len(prompt):].strip()
                else:
                    result_str = result_str.strip()

                elapsed = time.time() - start_time
                logger.info(f"LLM call completed in {elapsed:.2f}s")
                return result_str
                
            except OverflowError as e:
                logger.warning(f"OverflowError on attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    logger.info("Retrying with reduced max_tokens...")
                    max_tokens = max(10, max_tokens // 2)  # Reduce token limit by half, min 10
                    continue
                else:
                    logger.error(f"OverflowError persisted after {max_retries + 1} attempts")
                    return "Error: Unable to generate response due to tokenization issue. Please try rephrasing your query."
            
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise
        
        return "Error: Maximum retries exceeded."
