# llm_client.py

import asyncio
import logging
from config import Config
from models import SubPromptResponse
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BaseLLMClient:
    async def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, model: str):
        import google.generativeai as genai
        if not Config.GENIUS_API_KEY:
            raise ValueError("GENIUS_API_KEY environment variable not set")
        genai.configure(api_key=Config.GENIUS_API_KEY)
        self.model = genai.GenerativeModel(model)

    async def generate_content(self, prompt: str, stream: bool = False, generation_config=None):
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config = generation_config
            ))
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content from LLM: {e}")
            raise

# Singleton pattern for LLM client
_llm_client_instance = None

def get_llm_client() -> BaseLLMClient:
    global _llm_client_instance
    if _llm_client_instance is None:
        if Config.LLM_PROVIDER == "gemini":
            _llm_client_instance = GeminiLLMClient(model=Config.GENIUS_MODEL)
        else:
            raise ValueError(f"Unsupported LLM provider: {Config.LLM_PROVIDER}")
    return _llm_client_instance