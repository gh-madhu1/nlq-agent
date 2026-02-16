import unittest
from unittest.mock import MagicMock, patch
from utils.llm_client import LLMClient
import utils.llm_client as llm_module


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Clear the singleton cache before each test."""
        llm_module._llm_cache.clear()
    @patch('utils.llm_client.HuggingFacePipeline.from_model_id')
    @patch('utils.llm_client._detect_device')
    def test_local_model_initialization_mps(self, mock_detect, mock_pipeline):
        """Test local model init path for MPS (Apple Silicon)."""
        import torch
        mock_detect.return_value = ("mps", torch.float16)

        client = LLMClient(model_provider="local", model_name="Qwen/Qwen2.5-1.5B-Instruct")

        args, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs['model_id'], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(kwargs['task'], "text-generation")
        self.assertFalse(kwargs['pipeline_kwargs']['return_full_text'])
        self.assertTrue(kwargs['model_kwargs']['low_cpu_mem_usage'])
        self.assertTrue(kwargs['model_kwargs']['use_cache'])
        # MPS should NOT have quantization_config
        self.assertNotIn('quantization_config', kwargs['model_kwargs'])
        self.assertEqual(kwargs['model_kwargs']['torch_dtype'], torch.float16)
        self.assertIsNotNone(client.get_llm())

    @patch('utils.llm_client.HuggingFacePipeline.from_model_id')
    @patch('utils.llm_client._detect_device')
    def test_local_model_initialization_cuda(self, mock_detect, mock_pipeline):
        """Test local model init path for CUDA — should use quantization."""
        import torch
        mock_detect.return_value = ("cuda", torch.float16)

        client = LLMClient(model_provider="local", model_name="Qwen/Qwen2.5-1.5B-Instruct")

        args, kwargs = mock_pipeline.call_args
        self.assertIn('quantization_config', kwargs['model_kwargs'])

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
