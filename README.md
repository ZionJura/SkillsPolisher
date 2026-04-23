# SkillsPolisher

Research on skill-invocation improvement for LLM agents, combining:
- **CASE** — bandit-based demonstration selection for few-shot prompting
- **BPO** — black-box prompt optimization / rewriting
- **SPLICE** — skill-aware prompt selection and invocation engine
- Evaluated on **SkillsBench** and 6 reasoning / QA benchmarks

---

## Quick Start

### 1. Clone

```bash
git clone git@github.com:ZionJura/SkillsPolisher.git
cd SkillsPolisher
```

### 2. Install dependencies

```bash
bash scripts/setup.sh
```

### 3. Download datasets

```bash
python scripts/download_datasets.py              # all datasets (~500 MB)
python scripts/download_datasets.py --dataset gsm8k,skillsbench   # selective
```

### 4. Verify setup

```bash
python scripts/verify_setup.py           # full check (includes API ping)
python scripts/verify_setup.py --no-api  # skip API connectivity test
```

### 5. Run evaluation

```bash
# Single baseline, mock LLM (offline)
python eval_pipeline/run_eval.py --dataset gsm8k --baseline zero_shot --n_samples 50 --mock

# Compare all baselines
python eval_pipeline/run_compare.py --dataset gsm8k --n_samples 50 --mock

# Real API
python eval_pipeline/run_compare.py --dataset skillsbench --n_samples 88
```

---

## Project Structure

```
SkillsPolisher/
├── eval_pipeline/            # Unified eval pipeline
│   ├── run_eval.py           # Single-baseline evaluation
│   ├── run_compare.py        # Multi-baseline comparison table
│   ├── llm_client.py         # zhizengzeng / OpenAI-compatible LLM client
│   ├── configs/
│   │   └── eval_config.json  # API key, model, endpoint config
│   ├── datasets/
│   │   ├── data_utils.py     # Centralized path resolver (NEW)
│   │   ├── data/             # Downloaded datasets (after download_datasets.py)
│   │   ├── base.py           # EvalDataset + EvalSample base classes
│   │   ├── aqua_rat.py       # AQUA-RAT loader
│   │   ├── gsm8k.py          # GSM8K loader
│   │   ├── tabmwp.py         # TabMWP loader
│   │   ├── finqa.py          # FinQA loader
│   │   ├── strategyqa.py     # StrategyQA loader
│   │   ├── bpo.py            # BPO / Dolly / Self-Instruct loaders
│   │   ├── skillsbench.py    # SkillsBench loader
│   │   └── demo_bank.py      # Demo Bank loader
│   ├── baselines/
│   │   ├── zero_shot.py      # Zero-shot baseline
│   │   ├── random_kshot.py   # Random k-shot baseline
│   │   ├── case_bandit.py    # CASE bandit-based selection
│   │   └── bpo_rewrite.py    # BPO prompt rewriting
│   └── evaluators/           # Exact-match + LLM judge evaluators
├── splice/                   # SPLICE skill-invocation engine
│   ├── splice_loop.py        # Main SPLICE loop
│   ├── splice_select.py      # Demonstration selection
│   ├── splice_rewrite.py     # BPO-style rewriting
│   └── data/
│       └── demo_bank.json    # Skill invocation demonstrations
├── scripts/
│   ├── setup.sh              # Dependency installer
│   ├── download_datasets.py  # Dataset downloader
│   └── verify_setup.py       # End-to-end setup verifier
├── idea-stage/               # Research ideas and notes
└── related-works/            # Local checkouts (optional, legacy path support)
    ├── CASE/
    ├── BPO/
    └── skillsbench/
```

---

## Datasets

| Name | Source | Approx. Size | Description |
|------|--------|-------------|-------------|
| `aqua_rat` | HuggingFace `deepmind/aqua_rat` | 254 dev / 98k train | Algebraic word problems, MCQ |
| `gsm8k` | HuggingFace `openai/gsm8k` | 1,319 test / 7,473 train | Grade-school math, chain-of-thought |
| `tabmwp` | GitHub lupantech/PromptPG | 7,686 test | Table-based math word problems |
| `finqa` | GitHub czyssrs/FinQA | ~8,600 test | Financial QA with tables and text |
| `strategyqa` | HuggingFace `wics/strategy-qa` | ~490 test | Multi-hop yes/no reasoning |
| `bpo_test` | GitHub THUDM/BPO | 200 | Prompt-optimization benchmark |
| `dolly_eval` | GitHub THUDM/BPO | 200 | Dolly instruction-following eval |
| `self_instruct_eval` | GitHub THUDM/BPO | 252 | Self-instruct instruction following |
| `skillsbench` | GitHub benchflow-ai/skillsbench | 89 tasks | Skill-invocation agent benchmark |
| `demo_bank` | `splice/data/` (repo) | 88 demos | Skill invocation demonstration bank |

---

## Baselines

| Name | Description |
|------|-------------|
| `zero_shot` | Direct question → LLM, no demonstrations |
| `random_kshot` | k randomly sampled demonstrations from train set |
| `case_bandit` | CASE: bandit-based adaptive demonstration selection |
| `bpo_rewrite` | BPO: black-box prompt rewriting before inference |

---

## API Configuration

The pipeline defaults to the **zhizengzeng** OpenAI-compatible proxy, which routes to GPT-4o, Claude, Gemini, etc. Configure in `eval_pipeline/configs/eval_config.json`:

```json
{
  "llm": {
    "api_key": "sk-...",
    "base_url": "https://api.zhizengzeng.com/v1",
    "model": "gpt-4o-2024-11-20"
  }
}
```

Model tier shortcuts (set `--model` in run scripts):

| Alias | Model |
|-------|-------|
| `fast` | `gpt-4o-mini` |
| `standard` | `gpt-4o-2024-11-20` |
| `strong` | `gpt-4.1` |
| `reasoning` | `o4-mini` |
| `claude` | `claude-sonnet-4-6` |

Use `--mock` for fully offline testing (no API calls).

---

## Dataset Path Resolution

Dataset loaders resolve paths in this order:

1. `eval_pipeline/datasets/data/<name>/` — canonical location after `download_datasets.py`
2. `related-works/<legacy-path>/` — backward compatibility if you have a local `related-works/` checkout
3. Raises `DataNotFoundError` with instructions to run the download script

The resolver lives in `eval_pipeline/datasets/data_utils.py`.

---

## SPLICE System

SPLICE (Skill-Prompted LLM Invocation via Contextual Examples) is in `splice/`:

- **`splice_select.py`** — selects the most relevant skill demonstrations for a given task
- **`splice_rewrite.py`** — rewrites prompts using BPO-style optimization
- **`splice_loop.py`** — end-to-end loop combining selection, rewriting, and evaluation
- **`data/demo_bank.json`** — 88 skill invocation demonstrations covering SkillsBench tasks

Run SPLICE evaluation:
```bash
python splice/run_full_eval.py --dataset skillsbench --n_samples 20 --mock
```

---

## Development Notes

- All scripts must be run from the project root (the directory containing `eval_pipeline/`).
- Results are saved to `eval_pipeline/results/` as JSON files.
- The `--mock` flag uses a deterministic mock LLM — safe for CI and offline development.
