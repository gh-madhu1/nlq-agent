import unittest
from unittest.mock import MagicMock, patch
from utils.llm_client import LLMClient
import utils.llm_client as llm_module


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Clear the singleton cache before each test."""
        llm_module._llm_cache.clear()
    @patch('utils.llm_client.hf_pipeline')
    @patch('utils.llm_client.AutoModelForCausalLM.from_pretrained')
    @patch('utils.llm_client.AutoTokenizer.from_pretrained')
    @patch('utils.llm_client.GenerationConfig.from_pretrained')
    @patch('utils.llm_client._detect_device')
    def test_local_model_initialization_mps(self, mock_detect, mock_gen_config, mock_tokenizer, mock_model, mock_pipeline):
        """Test local model init path for MPS (Apple Silicon)."""
        import torch
        mock_detect.return_value = ("mps", torch.float16)
        
        # Setup mocks to avoid real loading
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_gen_config.return_value = MagicMock()

        client = LLMClient(model_provider="local", model_name="meta-llama/Llama-3.2-3B-Instruct")

        # Verify tokenizer was loaded
        mock_tokenizer.assert_called_once_with("meta-llama/Llama-3.2-3B-Instruct")
        
        # Verify model was loaded with correct kwargs
        args, kwargs = mock_model.call_args
        self.assertEqual(kwargs['torch_dtype'], torch.float16)
        self.assertEqual(kwargs['device_map'], 'mps')

        # Verify pipeline was created
        self.assertTrue(mock_pipeline.called)
        self.assertIsNotNone(client.get_llm())

    @patch('utils.llm_client.hf_pipeline')
    @patch('utils.llm_client.AutoModelForCausalLM.from_pretrained')
    @patch('utils.llm_client.AutoTokenizer.from_pretrained')
    @patch('utils.llm_client.GenerationConfig.from_pretrained')
    @patch('utils.llm_client._detect_device')
    def test_local_model_initialization_cuda(self, mock_detect, mock_gen_config, mock_tokenizer, mock_model, mock_pipeline):
        """Test local model init path for CUDA — should use quantization."""
        import torch
        mock_detect.return_value = ("cuda", torch.float16)
        
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_gen_config.return_value = MagicMock()

        client = LLMClient(model_provider="local", model_name="meta-llama/Llama-3.2-3B-Instruct")

        args, kwargs = mock_model.call_args
        self.assertIn('quantization_config', kwargs)

    @patch('utils.llm_client.ChatOpenAI')
    @patch('os.getenv')
    def test_openai_initialization(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "sk-test-key"
        client = LLMClient(model_provider="openai")
        self.assertIsNotNone(client.get_llm())

    @patch('utils.llm_client.ChatOpenAI')
    @patch('os.getenv')
    def test_call_strips_echoed_prompt(self, mock_getenv, mock_openai):
        """Test that prompt echo is stripped from the response."""
        mock_getenv.return_value = "sk-test-key"
        mock_llm = mock_openai.return_value
        client = LLMClient(model_provider="openai")

        # Simulate a response that echoes the prompt
        prompt = "What is 2+2?"
        mock_llm.invoke.return_value = prompt + " The answer is 4."
        result = client.call(prompt)
        self.assertEqual(result, "The answer is 4.")

    @patch('utils.llm_client.ChatOpenAI')
    @patch('os.getenv')
    def test_call_handles_content_attribute(self, mock_getenv, mock_openai):
        mock_getenv.return_value = "sk-test-key"
        mock_llm = mock_openai.return_value
        client = LLMClient(model_provider="openai")

        mock_response = MagicMock()
        mock_response.content = "Response Object"
        mock_llm.invoke.return_value = mock_response
        self.assertEqual(client.call("test"), "Response Object")


if __name__ == "__main__":
    unittest.main()
