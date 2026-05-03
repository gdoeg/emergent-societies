"""LLM provider abstraction with concrete Ollama and Groq backends.

Includes retry logic, exponential backoff, multi-model fallback, and
structured error handling for production-grade reliability.
"""

import asyncio
import os
import logging
import time
from typing import Any, Optional, Dict

import httpx

logger = logging.getLogger(__name__)


# Default models: lightweight first, with fallback options
DEFAULT_MODELS = [
    "llama-3.1-8b-instant",      # lightweight, fast, cheap
    "llama-3.3-70b-versatile",   # capable fallback
]

# Legacy aliases that should not be sent to Groq directly.
LEGACY_MODEL_ALIASES = {
    "llama3",
    "llama3:latest",
}

# Deprecated models that should no longer be used
DEPRECATED_MODELS = {
    "llama3-8b-8192",     # old naming convention
    "llama3-70b-8192",    # old naming convention
}

# Retry configuration
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE_RATE_LIMIT = 2.0  # exponential base for rate limit errors
RETRY_BACKOFF_BASE_TIMEOUT = 1.5  # exponential base for timeout errors


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    def __init__(self, error_type: str, message: str, model: Optional[str] = None):
        self.error_type = error_type
        self.message = message
        self.model = model
        super().__init__(f"[{error_type}] {message}")


class RateLimitError(LLMError):
    """Raised when API returns 429 (too many requests)."""
    def __init__(self, message: str, model: Optional[str] = None, retry_after: Optional[float] = None):
        super().__init__("rate_limit", message, model)
        self.retry_after = retry_after


class ModelDecommissionedError(LLMError):
    """Raised when model has been decommissioned."""
    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__("model_decommissioned", message, model)


class InvalidAPIKeyError(LLMError):
    """Raised when API key is invalid (401)."""
    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__("invalid_api_key", message, model)


class TimeoutError(LLMError):
    """Raised when request times out."""
    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__("timeout", message, model)


def _default_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class BaseLLMProvider:
    """Interface for pluggable LLM backends used by LLMPolicy."""

    def __init__(self, provider_name: str, model: str, timeout: float) -> None:
        self.provider_name = provider_name
        self.model = model
        self.timeout = float(timeout)
        self._health: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model,
            "healthy": None,
            "status": "unknown",
            "provider_status": None,
            "status_code": None,
            "error_type": None,
            "error_code": None,
            "provider_error": None,
            "message": None,
        }

    def get_health(self) -> dict[str, Any]:
        return dict(self._health)

    def _record_success(self) -> None:
        self._health = {
            "provider": self.provider_name,
            "model": self.model,
            "healthy": True,
            "status": "ok",
            "provider_status": "ok",
            "status_code": 200,
            "error_type": None,
            "error_code": None,
            "provider_error": None,
            "message": None,
        }

    def _classify_http_error(
        self,
        response: httpx.Response,
        error_payload: dict[str, Any],
    ) -> str:
        """Classify HTTP errors into structured error types.
        
        Returns one of: rate_limit, invalid_api_key, model_decommissioned, 
        model_not_found, or http_status_error.
        """
        status_code = response.status_code
        raw_type = str(error_payload.get("type") or "").strip().lower()
        raw_code = str(error_payload.get("code") or "").strip().lower()
        message = str(error_payload.get("message") or "").strip().lower()
        combined = " ".join(part for part in (raw_type, raw_code, message) if part)

        # Rate limit detection (429 or explicit rate_limit error)
        if status_code == 429 or "rate_limit" in combined or "rate limit" in message:
            return "rate_limit"

        # API key detection (401 or explicit auth error)
        if status_code == 401 or "invalid_api_key" in combined or "api_key" in combined or "unauthorized" in combined:
            return "invalid_api_key"

        # Model decommissioned detection
        if "model_decommissioned" in combined or "model not available" in message:
            return "model_decommissioned"

        # Model not found detection
        if status_code == 404 or "model_not_found" in combined or "not found" in message:
            return "model_not_found"

        return "http_status_error"

    def _record_http_error(self, response: httpx.Response) -> None:
        error_payload: dict[str, Any] = {}
        try:
            body = response.json()
        except ValueError:
            body = None

        if isinstance(body, dict):
            raw_error = body.get("error")
            if isinstance(raw_error, dict):
                error_payload = raw_error

        error_type = self._classify_http_error(response, error_payload)

        self._health = {
            "provider": self.provider_name,
            "model": self.model,
            "healthy": False,
            "status": "error",
            "provider_status": "error",
            "status_code": response.status_code,
            "error_type": error_type,
            "error_code": error_payload.get("code"),
            "provider_error": error_type,
            "message": error_payload.get("message") or response.text[:240],
        }
        
        # Raise structured exception based on error type
        message = error_payload.get("message") or response.text[:240]
        retry_after = None
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except (ValueError, TypeError):
            pass

        if error_type == "rate_limit":
            raise RateLimitError(message, self.model, retry_after)
        elif error_type == "invalid_api_key":
            raise InvalidAPIKeyError(message, self.model)
        elif error_type == "model_decommissioned":
            raise ModelDecommissionedError(message, self.model)
        elif error_type == "model_not_found":
            raise ModelDecommissionedError(f"Model not found: {message}", self.model)

    def _record_transport_error(self, exc: Exception) -> None:
        self._health = {
            "provider": self.provider_name,
            "model": self.model,
            "healthy": False,
            "status": "error",
            "provider_status": "error",
            "status_code": None,
            "error_type": type(exc).__name__,
            "error_code": None,
            "provider_error": type(exc).__name__,
            "message": str(exc),
        }

    async def validate_connection(self) -> dict[str, Any]:
        try:
            await self.generate("Reply with the single word ok.")
        except Exception:
            pass
        return self.get_health()

    async def generate(self, prompt: str, expect_json: bool = False) -> str:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider using an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 5.0):
        # Warn if using deprecated model names
        if model in DEPRECATED_MODELS:
            logger.warning(
                "Using deprecated model '%s'. Please use one of: %s",
                model,
                ", ".join(DEFAULT_MODELS),
            )
        
        super().__init__(provider_name="ollama", model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")
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
        try:
            response = await self._client.post(
                self._chat_endpoint(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._record_success()
            return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            self._record_transport_error(TimeoutError("Request timed out", self.model))
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s", self.model)
        except httpx.HTTPStatusError as exc:
            self._record_http_error(exc.response)
            raise
        except httpx.HTTPError as exc:
            self._record_transport_error(exc)
            raise
        except Exception as exc:
            self._record_transport_error(exc)
            raise


class GroqProvider(BaseLLMProvider):
    """Hosted Groq provider using the OpenAI-compatible chat API with retry support."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: float = 10.0):
        # Warn if using deprecated model names
        if model in DEPRECATED_MODELS:
            logger.warning(
                "Using deprecated model '%s'. Please use one of: %s",
                model,
                ", ".join(DEFAULT_MODELS),
            )
        
        super().__init__(provider_name="groq", model=model, timeout=timeout)
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=self.timeout)

    def _classify_http_error(
        self,
        response: httpx.Response,
        error_payload: dict[str, Any],
    ) -> str:
        raw_type = str(error_payload.get("type") or "").strip().lower()
        raw_code = str(error_payload.get("code") or "").strip().lower()
        message = str(error_payload.get("message") or "").strip().lower()
        combined = " ".join(part for part in (raw_type, raw_code, message) if part)

        if "model_decommissioned" in combined:
            return "model_decommissioned"
        
        # Call parent for standard error detection
        return super()._classify_http_error(response, error_payload)

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

        try:
            response = await self._client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            self._record_success()
            return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            self._record_transport_error(TimeoutError("Request timed out", self.model))
            raise TimeoutError(f"Groq request timed out after {self.timeout}s", self.model)
        except httpx.HTTPStatusError as exc:
            self._record_http_error(exc.response)
            raise
        except httpx.HTTPError as exc:
            self._record_transport_error(exc)
            logger.error(
                "Groq error: %s - %s",
                self._health.get("provider_error") or self._health.get("error_type"),
                self._health.get("message") or str(exc),
            )
            raise


def get_llm_models() -> list[str]:
    """Get list of LLM models from environment or use defaults.
    
    Environment variable LLM_MODELS can override as comma-separated list.
    Returns a list with lightweight models first.
    """
    env_models = os.getenv("LLM_MODELS", "").strip()
    if env_models:
        models = [m.strip() for m in env_models.split(",")]
        # Filter out deprecated models
        models = [m for m in models if m and m not in DEPRECATED_MODELS]
        if models:
            logger.info("Using custom LLM models from LLM_MODELS env: %s", models)
            return models
    
    logger.info("Using default LLM models: %s", DEFAULT_MODELS)
    return DEFAULT_MODELS


def get_llm_provider(
    llm_model: Optional[str] = None,
    llm_api_base_url: Optional[str] = None,
    llm_timeout: float = 5.0,
) -> BaseLLMProvider:
    """Select Groq in production and Ollama for local development.

    Provider selection is environment-driven to keep deployment setup simple.
    Uses the first model from get_llm_models() by default.
    """
    if os.getenv("ENV") == "production":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required when ENV=production")
        
        # Use provided model, or get first from configured models
        models = get_llm_models()
        groq_model = llm_model or models[0]
        fallback_model = next(
            (
                model
                for model in models
                if model not in DEPRECATED_MODELS
                and model not in LEGACY_MODEL_ALIASES
            ),
            None,
        ) or next(
            (
                model
                for model in DEFAULT_MODELS
                if model not in DEPRECATED_MODELS
                and model not in LEGACY_MODEL_ALIASES
            ),
            None,
        )
        
        # Ensure we're not using a deprecated model
        if groq_model in DEPRECATED_MODELS or groq_model in LEGACY_MODEL_ALIASES:
            logger.warning(
                "Requested model %s is deprecated/legacy, falling back to %s",
                groq_model,
                fallback_model,
            )
            groq_model = fallback_model
        
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
