# SPLICE

**SPLICE: Sample-Efficient Skill Invocation via Bandit-Selected ICL Demonstrations and Black-Box Prompt Rewriting**

SPLICE improves LLM agent skill invocation on SkillsBench by combining:
1. **SPLICE-Select**: CASE-style gap-index bandit for selecting ICL demonstrations
2. **SPLICE-Rewrite**: BPO-style black-box prompt rewriting for skill invocation prompts

## Setup

```bash
# Install dependencies
pip install numpy torch transformers openai anthropic benchflow>=0.3.0a7

# Set API keys (at least one required for non-dry-run mode)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Quick Start

### Pilot experiment (10 tasks, dry-run mode)
```bash
cd splice/
python run_pilot.py --eval_mode dry_run
```

### Pilot with real OpenAI evaluation
```bash
python run_pilot.py --eval_mode openai --rewriter_mode openai
```

### Pilot with real bench CLI + GPU BPO model
```bash
python run_pilot.py --eval_mode bench_cli --rewriter_mode hf
```

### Single task evaluation
```bash
python splice_loop.py --task 3d-scan-calc --method splice_full --eval_mode dry_run
```

### Full evaluation on all 89 tasks
```bash
python run_full_eval.py --eval_mode dry_run --methods baseline case_only splice_full
python run_full_eval.py --eval_mode bench_cli --n_workers 4  # parallel with bench CLI
```

## Files

| File | Description |
|------|-------------|
| `demo_bank.py` | Builds and manages (task, skill_invocation, outcome) demo bank |
| `splice_select.py` | CASE-style gap-index bandit for demo selection (`CASEBandit`) |
| `splice_rewrite.py` | BPO-style skill prompt rewriter (HF + OpenAI + Claude backends) |
| `eval_runner.py` | SkillsBench evaluation integration (bench CLI / OpenAI / dry-run) |
| `splice_loop.py` | Main SPLICE pipeline orchestration (`SPLICEPipeline`) |
| `run_pilot.py` | Pilot experiment: 10 tasks, 3 methods, <1 hour |
| `run_full_eval.py` | Full evaluation on all 89 tasks, parallel support |
| `configs/splice_default.json` | Default hyperparameters |

## Methods

| Method | Description |
|--------|-------------|
| `baseline` | No demos, original skill prompt |
| `random_kshot` | Random k demos, original prompt |
| `bpo_only` | BPO-rewritten prompt, random demos (no bandit) |
| `case_only` | CASE bandit-selected demos, original prompt |
| `splice_select` | Alias for case_only |
| `splice_rewrite` | CASE demos + BPO-rewritten prompt |
| `splice_full` | Full SPLICE: bandit selection + BPO rewriting (iterative) |

## Evaluation Backends

| Mode | Description | Requirement |
|------|-------------|-------------|
| `dry_run` | Mock evaluation (fast, for testing) | None |
| `openai` | GPT-based task evaluation | `OPENAI_API_KEY` |
| `bench_cli` | Real SkillsBench via `bench eval create` | `pip install benchflow` + agent API key |

## Rewriter Backends

| Mode | Description | Requirement |
|------|-------------|-------------|
| `auto` | Auto-selects: GPU→HF, else OpenAI, else Claude | Whichever is available |
| `hf` | `THUDM/BPO` HuggingFace model (4-bit quantized) | GPU + transformers |
| `openai` | GPT-4o-mini API | `OPENAI_API_KEY` |
| `claude` | Claude API | `ANTHROPIC_API_KEY` |

## Configuration

Edit `configs/splice_default.json` or pass `--config my_config.json`.

Key hyperparameters:
- `bandit.delta`: Confidence level for CASE bandit (default: 0.05)
- `bandit.k_selected`: Number of demos to select (default: 3)
- `bandit.max_rounds`: Maximum bandit rounds (default: 100)
- `rewriter.model_id`: BPO model (default: `THUDM/BPO`)
- `loop.n_rounds`: Iterative SPLICE rounds (default: 1)

## Expected Results (Table 1)

| Method | Expected Success Rate |
|--------|----------------------|
| baseline | ~35-40% |
| random_kshot | ~38-42% |
| bpo_only | ~42-46% |
| case_only | ~45-50% |
| splice_full | ~52-58% |
| oracle | ~85-90% |
