"""
llm_client.py — Centralized LLM Client for Eval Pipeline

Uses zhizengzeng OpenAI-compatible API (standard openai package, base_url pattern).
See: shared-references/default-api-router.md for full model list and routing rules.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default API (zhizengzeng) ─────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_API_KEY  = "sk-zk218fe5b9b393541e35b39f29f083a22ec69c91a6228740"
DEFAULT_MODEL    = "gpt-4o-2024-11-20"

# Model tier shortcuts
MODEL_TIERS = {
    "fast":      "gpt-4o-mini",
    "standard":  "gpt-4o-2024-11-20",
    "strong":    "gpt-4.1",
    "reasoning": "o4-mini",
    "best":      "gpt-5.4-pro",
    "claude":    "claude-sonnet-4-6",
    "gemini":    "gemini-2.5-pro",
    "deepseek":  "deepseek-v3",
}


class CostTracker:
    """Tracks token usage and call counts across LLM calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.n_calls = 0

    def update(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.n_calls += 1

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "n_calls": self.n_calls,
        }

    def reset(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.n_calls = 0


class LLMClient:
    """
    Centralized LLM client using the zhizengzeng OpenAI-compatible API.

    Args:
        model:      Model name or tier alias (fast/standard/strong/reasoning/best).
                    Defaults to DEFAULT_MODEL (gpt-4o-2024-11-20).
        base_url:   API base URL. Defaults to https://api.zhizengzeng.com/v1.
        api_key:    API key. Defaults to zhizengzeng key.
        mock_mode:  If True, returns '[MOCK RESPONSE]' without any API calls.
        max_retries: Max retry attempts on rate limit / server errors.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
        max_retries: int = 3,
    ):
        raw_model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        # Resolve tier aliases (e.g. "fast" → "gpt-4o-mini")
        self.model    = MODEL_TIERS.get(raw_model, raw_model)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key  = api_key  or os.getenv("LLM_API_KEY",  DEFAULT_API_KEY)
        self.mock_mode   = mock_mode
        self.max_retries = max_retries
        self.cost_tracker = CostTracker()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from e
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def chat(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages:   List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Response content string.

        Raises:
            ConnectionError: If network / API is unreachable.
            RuntimeError:    If all retries are exhausted.
        """
        if self.mock_mode:
            est_prompt = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            self.cost_tracker.update(est_prompt, 10)
            return "[MOCK RESPONSE]"

        client = self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = response.usage
                if usage:
                    self.cost_tracker.update(usage.prompt_tokens, usage.completion_tokens)
                else:
                    est = sum(len(str(m.get("content", ""))) // 4 for m in messages)
                    self.cost_tracker.update(est, max_tokens // 4)

                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                etype = type(e).__name__

                # Unrecoverable: network unreachable
                if "APIConnectionError" in etype or "ConnectError" in etype:
                    raise ConnectionError(
                        f"API endpoint unreachable: {self.base_url}\n"
                        f"Check network connectivity. Error: {e}"
                    ) from e

                # Recoverable: rate limit / server errors — retry with backoff
                if any(x in etype for x in ("RateLimit", "InternalServer",
                                             "ServiceUnavailable", "APIStatus")):
                    wait = 2.0 ** attempt
                    logger.warning(
                        f"API error attempt {attempt+1}/{self.max_retries}: {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    last_error = e
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
                    continue

                raise RuntimeError(f"LLM API error: {e}") from e

        raise RuntimeError(
            f"LLM API failed after {self.max_retries} retries. Last: {last_error}"
        )

    @classmethod
    def from_config(cls, config: dict, mock_mode: bool = False) -> "LLMClient":
        """Create LLMClient from config dict (loaded from eval_config.json)."""
        llm_cfg = config.get("llm", config)
        # Support tier alias in config
        model = llm_cfg.get("model", DEFAULT_MODEL)
        model = MODEL_TIERS.get(model, model)
        return cls(
            model=model,
            base_url=llm_cfg.get("base_url", DEFAULT_BASE_URL),
            api_key=llm_cfg.get("api_key", DEFAULT_API_KEY),
            mock_mode=mock_mode,
        )

    def reset_cost_tracker(self) -> None:
        self.cost_tracker.reset()

    def __repr__(self) -> str:
        mode = "mock" if self.mock_mode else f"zhizengzeng/{self.model}"
        return f"LLMClient(mode={mode!r}, calls={self.cost_tracker.n_calls})"
