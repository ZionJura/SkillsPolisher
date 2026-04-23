"""
splice_rewrite.py — BPO-Style Skill Prompt Rewriter

Takes (skill_prompt, selected_demos) and produces a rewritten skill
invocation prompt using either:
  - Mode "hf": THUDM/BPO HuggingFace model (7B LLaMA-based rewriter)
  - Mode "openai": OpenAI API fallback (GPT-4o-mini by default)

BPO reference: Cheng et al., ACL 2024
Model: https://huggingface.co/THUDM/BPO

Usage:
    rewriter = SkillPromptRewriter(mode="hf")          # or mode="openai"
    rewritten = rewriter.rewrite(skill_prompt, selected_demos)
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── BPO prompt template (from BPO/src/infer_example.py) ─────────────────────

# Original BPO template (general prompt rewriting)
BPO_TEMPLATE_GENERAL = (
    "[INST] You are an expert prompt engineer. "
    "Please help me improve this prompt to get a more helpful and harmless response:\n"
    "{prompt} [/INST]"
)

# SPLICE-specific template: skill-invocation aware
SPLICE_TEMPLATE = (
    "[INST] You are an expert at writing skill invocation prompts for AI agents. "
    "The following are examples of successful skill invocations:\n\n"
    "{demos}\n\n"
    "Based on these examples, please rewrite the following skill instruction to be "
    "clearer, more actionable, and more likely to succeed:\n\n"
    "{skill_prompt}\n\n"
    "Provide only the improved skill instruction, without explanation. [/INST]"
)

# OpenAI system message for fallback
OPENAI_SYSTEM_MSG = (
    "You are an expert at writing skill invocation prompts for AI agents. "
    "You improve skill instructions to make them clearer, more actionable, and "
    "more likely to lead to successful task completion."
)

OPENAI_USER_TEMPLATE = (
    "Here are examples of successful skill invocations:\n\n"
    "{demos}\n\n"
    "Please rewrite the following skill instruction to be clearer and more effective:\n\n"
    "{skill_prompt}\n\n"
    "Provide only the improved instruction, without explanation."
)


# ── Helper: format demos for prompt ─────────────────────────────────────────

def _format_demos(demos: List[Dict[str, Any]], max_demos: int = 3, max_chars_each: int = 300) -> str:
    """
    Format selected demonstrations as few-shot examples in the prompt.

    Args:
        demos: List of demo dicts from DemoBank.
        max_demos: Max number of demos to include.
        max_chars_each: Max characters per demo field.

    Returns:
        Formatted string of demonstrations.
    """
    lines = []
    for idx, demo in enumerate(demos[:max_demos]):
        instr = str(demo.get("instruction", ""))[:max_chars_each]
        invoc = str(demo.get("invocation", ""))[:max_chars_each]
        skill = str(demo.get("skill_name", ""))
        lines.append(
            f"Example {idx + 1}:\n"
            f"  Task: {instr.strip()}\n"
            f"  Skill: {skill}\n"
            f"  Invocation: {invoc.strip()}"
        )
    return "\n\n".join(lines) if lines else "(no examples available)"


# ── HuggingFace BPO model wrapper ────────────────────────────────────────────

class HFBPORewriter:
    """
    Wraps THUDM/BPO HuggingFace model for skill prompt rewriting.

    Args:
        model_id: HuggingFace model ID (default: THUDM/BPO).
        device: Device string (e.g., "cuda:0", "cpu").
        load_in_4bit: Use 4-bit quantization for GPU memory efficiency.
        load_in_8bit: Use 8-bit quantization (mutually exclusive with 4bit).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling p.
    """

    def __init__(
        self,
        model_id: str = "THUDM/BPO",
        device: str = "cuda",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for HF mode. "
                "Install with: pip install transformers torch"
            ) from e

        print(f"[BPO] Loading model {self.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, add_prefix_space=True
        )

        load_kwargs: Dict[str, Any] = {}
        if self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            import torch
            load_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **load_kwargs
        )
        if not (self.load_in_4bit or self.load_in_8bit):
            self._model = self._model.eval().to(self.device)
        else:
            self._model = self._model.eval()

        print(f"[BPO] Model loaded.")

    def rewrite(
        self,
        skill_prompt: str,
        selected_demos: Optional[List[Dict[str, Any]]] = None,
        use_splice_template: bool = True,
    ) -> str:
        """
        Rewrite skill_prompt using the BPO model.

        Args:
            skill_prompt: The skill invocation prompt to improve.
            selected_demos: Optional list of demo dicts for few-shot context.
            use_splice_template: If True, use SPLICE's skill-aware template.
                                 If False, use original BPO general template.

        Returns:
            Rewritten skill prompt string.
        """
        self._load()

        import torch

        # Build the prompt
        if use_splice_template and selected_demos:
            demos_text = _format_demos(selected_demos)
            input_text = SPLICE_TEMPLATE.format(
                demos=demos_text,
                skill_prompt=skill_prompt[:1000],
            )
        else:
            input_text = BPO_TEMPLATE_GENERAL.format(prompt=skill_prompt[:1000])

        model_inputs = self._tokenizer(input_text, return_tensors="pt")
        if not (self.load_in_4bit or self.load_in_8bit):
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                num_beams=1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode: strip the [/INST] split as in BPO infer_example.py
        full_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        if "[/INST]" in full_text:
            result = full_text.split("[/INST]")[1].strip()
        else:
            # Fallback: strip input tokens
            input_len = model_inputs["input_ids"].shape[1]
            result = self._tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()

        return result if result else skill_prompt


# ── OpenAI API fallback rewriter ─────────────────────────────────────────────

class OpenAIBPORewriter:
    """
    OpenAI API fallback for BPO-style skill prompt rewriting.

    Uses GPT-4o-mini by default. Requires OPENAI_API_KEY environment variable.

    Args:
        model: OpenAI model name.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        max_retries: Max API call retries on transient errors.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAI mode. "
                    "Install with: pip install openai"
                ) from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it or use mode='hf'."
                )
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def rewrite(
        self,
        skill_prompt: str,
        selected_demos: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Rewrite skill_prompt using OpenAI API.

        Args:
            skill_prompt: The skill invocation prompt to improve.
            selected_demos: Optional list of demo dicts for few-shot context.

        Returns:
            Rewritten skill prompt string.
        """
        client = self._get_client()
        demos_text = _format_demos(selected_demos) if selected_demos else "(no examples)"
        user_msg = OPENAI_USER_TEMPLATE.format(
            demos=demos_text,
            skill_prompt=skill_prompt[:2000],
        )

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": OPENAI_SYSTEM_MSG},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [warn] OpenAI API error (attempt {attempt+1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  [error] OpenAI API failed after {self.max_retries} attempts: {e}")
                    return skill_prompt  # fall back to original on persistent failure


# ── Anthropic Claude API fallback ────────────────────────────────────────────

class ClaudeBPORewriter:
    """
    Anthropic Claude API fallback for BPO-style prompt rewriting.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        model: Anthropic model name (default: claude-3-haiku-20240307 for speed).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for Claude mode. "
                    "Install with: pip install anthropic"
                ) from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def rewrite(
        self,
        skill_prompt: str,
        selected_demos: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Rewrite skill_prompt using Anthropic Claude API."""
        client = self._get_client()
        demos_text = _format_demos(selected_demos) if selected_demos else "(no examples)"
        user_msg = OPENAI_USER_TEMPLATE.format(
            demos=demos_text,
            skill_prompt=skill_prompt[:2000],
        )

        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=OPENAI_SYSTEM_MSG,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [warn] Claude API error (attempt {attempt+1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  [error] Claude API failed after {self.max_retries} attempts: {e}")
                    return skill_prompt


# ── Unified SkillPromptRewriter ───────────────────────────────────────────────

class SkillPromptRewriter:
    """
    Unified SPLICE-Rewrite component.

    Selects the appropriate backend based on mode and availability.

    Args:
        mode: "hf" (HuggingFace BPO), "openai" (OpenAI API),
              "claude" (Anthropic Claude), or "auto" (tries hf, falls back).
        hf_model_id: HuggingFace model ID for HF mode.
        openai_model: OpenAI model name.
        claude_model: Anthropic model name.
        load_in_4bit: 4-bit quantization for HF mode.
        max_new_tokens: Max generated tokens.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        mode: str = "auto",
        hf_model_id: str = "THUDM/BPO",
        openai_model: str = "gpt-4o-mini",
        claude_model: str = "claude-3-haiku-20240307",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.mode = mode
        self._backend = None
        self._backend_type = None

        self._hf_kwargs = dict(
            model_id=hf_model_id,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._openai_kwargs = dict(
            model=openai_model,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._claude_kwargs = dict(
            model=claude_model,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        if mode != "auto":
            self._init_backend(mode)

    def _init_backend(self, mode: str) -> None:
        """Initialize the specified backend."""
        if mode == "hf":
            self._backend = HFBPORewriter(**self._hf_kwargs)
            self._backend_type = "hf"
        elif mode == "openai":
            self._backend = OpenAIBPORewriter(**self._openai_kwargs)
            self._backend_type = "openai"
        elif mode == "claude":
            self._backend = ClaudeBPORewriter(**self._claude_kwargs)
            self._backend_type = "claude"
        elif mode == "mock":
            self._backend = None  # handled inline
            self._backend_type = "mock"
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: hf, openai, claude, mock, auto")

    def _auto_select(self) -> None:
        """Auto-select backend: try HF, then OpenAI, then Claude."""
        # Try HF first (requires GPU + transformers)
        try:
            import torch
            import transformers
            if torch.cuda.is_available():
                print("[BPO] GPU detected, attempting HuggingFace BPO model...")
                self._backend = HFBPORewriter(**self._hf_kwargs)
                self._backend_type = "hf"
                return
            else:
                print("[BPO] No GPU detected, falling back to API mode.")
        except ImportError:
            print("[BPO] transformers/torch not available, falling back to API mode.")

        # Try OpenAI API
        if os.environ.get("OPENAI_API_KEY"):
            print("[BPO] Using OpenAI API fallback.")
            self._backend = OpenAIBPORewriter(**self._openai_kwargs)
            self._backend_type = "openai"
            return

        # Try Anthropic Claude
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("[BPO] Using Anthropic Claude API fallback.")
            self._backend = ClaudeBPORewriter(**self._claude_kwargs)
            self._backend_type = "claude"
            return

        raise RuntimeError(
            "No rewrite backend available. Please either:\n"
            "  1. Install torch + transformers and ensure a GPU is available, or\n"
            "  2. Set OPENAI_API_KEY environment variable, or\n"
            "  3. Set ANTHROPIC_API_KEY environment variable."
        )

    def rewrite(
        self,
        skill_prompt: str,
        selected_demos: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Rewrite skill_prompt with BPO-style optimization.

        Args:
            skill_prompt: The skill invocation prompt to improve.
            selected_demos: Optional list of selected demo dicts.

        Returns:
            Rewritten (improved) skill prompt string.
        """
        if self._backend_type == "mock":
            # Mock mode: return slightly modified prompt (for dry-run testing)
            prefix = "Optimized: " if selected_demos else ""
            return prefix + skill_prompt[:500]

        if self._backend is None:
            self._auto_select()

        try:
            result = self._backend.rewrite(
                skill_prompt=skill_prompt,
                selected_demos=selected_demos or [],
            )
            return result if result.strip() else skill_prompt
        except Exception as e:
            print(f"  [warn] Rewrite backend error: {e}. Returning original prompt.")
            return skill_prompt

    @property
    def backend_type(self) -> Optional[str]:
        """Return the active backend type."""
        return self._backend_type

    def __repr__(self) -> str:
        return f"SkillPromptRewriter(mode={self.mode!r}, backend={self._backend_type!r})"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test SPLICE-Rewrite skill prompt rewriter")
    parser.add_argument("--mode", choices=["hf", "openai", "claude", "auto"], default="auto")
    parser.add_argument("--skill_prompt", type=str, default=None,
                        help="Skill prompt to rewrite (reads from stdin if not given)")
    parser.add_argument("--demo_bank", type=str, default=None,
                        help="Path to demo_bank.json for example demos")
    parser.add_argument("--k", type=int, default=3, help="Number of demo examples to use")
    args = parser.parse_args()

    # Load skill prompt
    if args.skill_prompt:
        skill_prompt = args.skill_prompt
    else:
        print("Enter skill prompt (Ctrl+D when done):")
        skill_prompt = "\n".join(iter(input, ""))

    # Optionally load demos
    selected_demos: List[Dict] = []
    if args.demo_bank:
        try:
            with open(args.demo_bank) as f:
                data = json.load(f) if hasattr(__import__("json"), "load") else []
            import json
            with open(args.demo_bank) as f:
                data = json.load(f)
            selected_demos = data.get("demos", [])[:args.k]
            print(f"Loaded {len(selected_demos)} demos from {args.demo_bank}")
        except Exception as e:
            print(f"Could not load demo bank: {e}")

    # Rewrite
    rewriter = SkillPromptRewriter(mode=args.mode)
    print(f"\n[Original prompt]\n{skill_prompt}\n")
    rewritten = rewriter.rewrite(skill_prompt, selected_demos)
    print(f"[Rewritten prompt]\n{rewritten}\n")
    print(f"Backend used: {rewriter.backend_type}")


if __name__ == "__main__":
    import json
    main()
