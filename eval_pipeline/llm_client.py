"""
llm_client.py — Centralized LLM Client for Eval Pipeline

Supports two API backends, selected automatically via env vars or constructor args:

  1. Standard OpenAI-compatible (default) — zhizengzeng, works from any machine:
       base_url  = https://api.zhizengzeng.com/v1
       api_key   = sk-zk...

  2. Azure OpenAI — ByteDance internal, requires ByteDance network/VPN:
       azure_endpoint  = https://aidp.bytedance.net/api/modelhub/online/v2/crawl
       api_key         = NmLlK0R...
       api_version     = 2024-02-01
       default_headers = {"X-TT-LOGID": "bytebrain.aiops.faultscout_cn_zr"}

Switch via env var:
    export LLM_BACKEND=azure          # use ByteDance
    export LLM_BACKEND=openai         # use zhizengzeng (default)

Or per-run:
    python eval_pipeline/run_eval.py --backend azure ...
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Backend: zhizengzeng (default, works everywhere) ─────────────────────────
OPENAI_BASE_URL = "https://api.zhizengzeng.com/v1"
OPENAI_API_KEY  = "sk-zk218fe5b9b393541e35b39f29f083a22ec69c91a6228740"

# ── Backend: ByteDance Azure OpenAI (internal network only) ──────────────────
AZURE_ENDPOINT    = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
AZURE_API_KEY     = "NmLlK0RvFZrb1MCC0ZuiZSVa3ysJUVEZ_GPT_AK"
AZURE_API_VERSION = "2024-02-01"
AZURE_HEADERS     = {"X-TT-LOGID": "bytebrain.aiops.faultscout_cn_zr"}

DEFAULT_MODEL = "gpt-4o-2024-11-20"

MODEL_ALIASES = {
    "gpt-4-o": "gpt-4o-2024-11-20",
}

# Model tier shortcuts (same names work on both backends)
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


def _resolve_model_name(raw_model: str) -> str:
    """Resolve tier aliases and user-facing model aliases."""
    tier_resolved = MODEL_TIERS.get(raw_model, raw_model)
    return MODEL_ALIASES.get(tier_resolved, tier_resolved)


def _infer_backend(raw_model: Optional[str], explicit_backend: Optional[str]) -> str:
    """
    Choose backend.

    Rules:
      1. Explicit backend arg/env wins.
      2. Requested model name `gpt-4-o` uses ByteDance Azure.
      3. Everything else defaults to zhizengzeng openai-compatible API.
    """
    if explicit_backend:
        return explicit_backend.lower()
    if raw_model and raw_model.strip().lower() == "gpt-4-o":
        return "azure"
    return "openai"


class CostTracker:
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
    Unified LLM client supporting both zhizengzeng and ByteDance Azure backends.

    Args:
        model:    Model name or tier alias (fast/standard/strong/reasoning/best).
        backend:  "openai" (zhizengzeng, default) or "azure" (ByteDance internal).
                  Also reads LLM_BACKEND env var if not specified.
        mock_mode: Returns '[MOCK RESPONSE]' without API calls (offline testing).
        max_retries: Retry count on rate-limit / server errors.

    Override individual fields via env vars:
        LLM_BACKEND      openai | azure
        LLM_MODEL        model name or tier alias
        LLM_BASE_URL     override base URL (openai backend only)
        LLM_API_KEY      override API key (either backend)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        mock_mode: bool = False,
        max_retries: int = 3,
        # Advanced overrides (rarely needed)
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        default_headers: Optional[dict] = None,
    ):
        raw_model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        explicit_backend = backend or os.getenv("LLM_BACKEND")
        self.requested_model = raw_model
        self.model = _resolve_model_name(raw_model)
        self.backend = _infer_backend(raw_model, explicit_backend)
        self.mock_mode = mock_mode
        self.max_retries = max_retries
        self.cost_tracker = CostTracker()
        self._client = None

        # Store overrides for lazy init
        if self.backend == "azure":
            self._base_url = None
            self._api_key = (
                api_key
                or os.getenv("AZURE_OPENAI_API_KEY")
                or os.getenv("LLM_API_KEY")
            )
            self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self._api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        else:
            self._base_url = (
                base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("LLM_BASE_URL")
            )
            self._api_key = (
                api_key
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("LLM_API_KEY")
            )
            self._azure_endpoint = azure_endpoint
            self._api_version = api_version
        self._default_headers = default_headers

    def _get_client(self):
        """Lazy-initialize the appropriate OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI, AzureOpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        if self.backend == "azure":
            self._client = AzureOpenAI(
                api_key       = self._api_key or AZURE_API_KEY,
                api_version   = self._api_version or AZURE_API_VERSION,
                azure_endpoint= self._azure_endpoint or AZURE_ENDPOINT,
                default_headers= self._default_headers or AZURE_HEADERS,
            )
        else:
            # Standard OpenAI-compatible (zhizengzeng)
            self._client = OpenAI(
                base_url = self._base_url or OPENAI_BASE_URL,
                api_key  = self._api_key  or OPENAI_API_KEY,
            )

        return self._client

    def chat(
        self,
        messages: list,
        max_tokens: int = 12048,
        temperature: float = 0.0,
    ) -> str:
        if self.mock_mode:
            est = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            self.cost_tracker.update(est, 10)
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

                if "APIConnectionError" in etype or "ConnectError" in etype:
                    endpoint = AZURE_ENDPOINT if self.backend == "azure" else OPENAI_BASE_URL
                    hint = (
                        "\nHint: ByteDance API requires internal network or VPN."
                        if self.backend == "azure"
                        else ""
                    )
                    raise ConnectionError(
                        f"API unreachable (backend={self.backend}): {endpoint}{hint}\n"
                        f"To switch backend: export LLM_BACKEND=openai"
                    ) from e

                if any(x in etype for x in ("RateLimit", "InternalServer",
                                             "ServiceUnavailable", "APIStatus")):
                    wait = 2.0 ** attempt
                    logger.warning(f"Retry {attempt+1}/{self.max_retries} in {wait:.1f}s: {e}")
                    last_error = e
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
                    continue

                raise RuntimeError(f"LLM API error: {e}") from e

        raise RuntimeError(f"LLM failed after {self.max_retries} retries. Last: {last_error}")

    @classmethod
    def from_config(cls, config: dict, mock_mode: bool = False) -> "LLMClient":
        """Create from config dict (eval_config.json). Backend auto-detected."""
        llm_cfg = config.get("llm", config)
        model = llm_cfg.get("model", DEFAULT_MODEL)
        backend = _infer_backend(model, llm_cfg.get("backend"))
        return cls(
            model=model,
            backend=backend,
            mock_mode=mock_mode,
            base_url=llm_cfg.get("base_url") if backend == "openai" else None,
            api_key=(
                llm_cfg.get("api_key")
                if backend == "openai"
                else llm_cfg.get("azure_api_key")
            ),
            azure_endpoint=llm_cfg.get("azure_endpoint") if backend == "azure" else None,
            api_version=llm_cfg.get("api_version") if backend == "azure" else None,
            default_headers=llm_cfg.get("default_headers"),
        )

    def reset_cost_tracker(self) -> None:
        self.cost_tracker.reset()

    def __repr__(self) -> str:
        if self.mock_mode:
            return f"LLMClient(mode='mock', calls={self.cost_tracker.n_calls})"
        return (
            f"LLMClient(backend={self.backend!r}, model={self.model!r}, "
            f"requested_model={self.requested_model!r}, calls={self.cost_tracker.n_calls})"
        )
