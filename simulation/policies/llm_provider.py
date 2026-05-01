"""LLM provider abstraction with concrete Ollama and Groq backends."""

import os
from typing import Optional

import httpx


class BaseLLMProvider:
    """Interface for pluggable LLM backends used by LLMPolicy."""

    async def generate(self, prompt: str, expect_json: bool = False) -> str:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider using an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    def _chat_endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def generate(self, prompt: str, expect_json: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0.0,
            # Keep model loaded in Ollama to avoid cold reload between calls.
            "keep_alive": -1,
        }
        if expect_json:
            payload["format"] = "json"

        response = await self._client.post(
            self._chat_endpoint(),
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class GroqProvider(BaseLLMProvider):
    """Hosted Groq provider using the OpenAI-compatible chat API."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192", timeout: float = 10.0):
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def generate(self, prompt: str, expect_json: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a decision-making agent."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        response = await self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]


def get_llm_provider(
    llm_model: Optional[str] = None,
    llm_api_base_url: Optional[str] = None,
    llm_timeout: float = 5.0,
) -> BaseLLMProvider:
    """Select Groq in production and Ollama for local development.

    Provider selection is environment-driven to keep deployment setup simple.
    """
    if os.getenv("ENV") == "production":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required when ENV=production")
        groq_model = llm_model if llm_model not in (None, "", "llama3") else "llama3-8b-8192"
        return GroqProvider(
            api_key=api_key,
            model=groq_model,
            timeout=max(10.0, float(llm_timeout)),
        )

    return OllamaProvider(
        base_url=llm_api_base_url or "http://localhost:11434/v1",
        model=llm_model or "llama3",
        timeout=float(llm_timeout),
    )
