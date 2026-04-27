# SPLICE Eval Pipeline

Unified evaluation pipeline for benchmarking SPLICE against baselines across multiple datasets.

## Setup

Run all commands from the project root.

```bash
pip install openai numpy
# Optional: pip install tomli  # if Python < 3.11
```

## Quick Start (Offline / Mock Mode)

Test the pipeline without API calls:

```bash
# Zero-shot on GSM8K, 10 samples, mock LLM
python eval_pipeline/run_eval.py --dataset gsm8k --split test \
    --baseline zero_shot --n_samples 10 --mock --verbose

# AQUA-RAT dev, random k-shot
python eval_pipeline/run_eval.py --dataset aqua_rat --split dev \
    --baseline random_kshot --k 3 --n_samples 10 --mock

# Compare all baselines on StrategyQA
python eval_pipeline/run_compare.py --dataset strategyqa --split test \
    --n_samples 10 --mock --model_name gpt-5.4-mini
```

## run_eval.py — Single Baseline Evaluation

```bash
python eval_pipeline/run_eval.py \
    --dataset <name> \
    --split <train|dev|test> \
    --baseline <zero_shot|random_kshot|case_bandit|bpo_rewrite> \
    --n_samples <N>         # 0 = all samples
    --k <K>                 # demos for k-shot (default: 3)
    --seed <S>              # random seed (default: 42)
    --output <path.json>    # output file (default: ./results/{model_name}/{dataset_name}/{baseline}_result.json)
    --model_name <model>    # gpt-4-o -> ByteDance API, others -> zhizengzeng
    --mock                  # offline testing (no API calls)
    --verbose               # print each prediction
```

### Per-dataset examples

```bash
# AQUA-RAT (254 dev samples)
python eval_pipeline/run_eval.py --dataset aqua_rat --split dev \
    --baseline zero_shot --n_samples 10 --mock

# GSM8K (1319 test samples)
python eval_pipeline/run_eval.py --dataset gsm8k --split test \
    --baseline random_kshot --k 3 --n_samples 10 --mock

# TabMWP (7686 test samples)
python eval_pipeline/run_eval.py --dataset tabmwp --split test \
    --baseline zero_shot --n_samples 10 --mock

# FinQA (~8600 test samples)
python eval_pipeline/run_eval.py --dataset finqa --split test \
    --baseline case_bandit --k 3 --n_samples 10 --mock

# StrategyQA (~490 test samples)
python eval_pipeline/run_eval.py --dataset strategyqa --split test \
    --baseline zero_shot --n_samples 10 --mock

# BPO Test (200 samples)
python eval_pipeline/run_eval.py --dataset bpo_test --split test \
    --baseline bpo_rewrite --n_samples 10 --mock

# Dolly Eval (200 samples)
python eval_pipeline/run_eval.py --dataset dolly_eval --split test \
    --baseline zero_shot --n_samples 10 --mock

# Self-Instruct Eval (252 samples)
python eval_pipeline/run_eval.py --dataset self_instruct_eval --split test \
    --baseline zero_shot --n_samples 10 --mock

# SkillsBench (89 tasks)
python eval_pipeline/run_eval.py --dataset skillsbench --split test \
    --baseline zero_shot --n_samples 10 --mock

# Demo Bank (88 demos)
python eval_pipeline/run_eval.py --dataset demo_bank --split train \
    --baseline zero_shot --n_samples 10 --mock
```

## run_compare.py — Multi-Baseline Comparison

```bash
python eval_pipeline/run_compare.py \
    --dataset gsm8k \
    --split test \
    --n_samples 10 \
    --model_name gpt-5.4-mini \
    --mock
```

Real API examples:

```bash
# ByteDance API via model alias
python eval_pipeline/run_compare.py --dataset skillsbench --n_samples 10 --model_name gpt-4-o

# zhizengzeng API
python eval_pipeline/run_compare.py --dataset skillsbench --n_samples 10 --model_name deepseek-v3
```

Output (console + JSON):
```
+--------------------+----------+--------------+--------------+------------+--------------+------------+
| baseline           | n        | accuracy     | avg_score    | n_calls    | tokens       | time_s     |
+--------------------+----------+--------------+--------------+------------+--------------+------------+
| zero_shot          | 100      | 72.0%        | 0.720        | 100        | 12500        | 15.2s      |
| random_kshot       | 100      | 75.0%        | 0.750        | 100        | 18000        | 16.1s      |
| case_bandit        | 100      | 77.0%        | 0.770        | 100        | 17500        | 22.3s      |
| bpo_rewrite        | 100      | 73.0%        | 0.730        | 100        | 19200        | 18.5s      |
+--------------------+----------+--------------+--------------+------------+--------------+------------+
```

## Output Format

Results JSON:
```json
{
  "dataset": "gsm8k",
  "split": "test",
  "baseline": "zero_shot",
  "n_samples": 100,
  "accuracy": 0.72,
  "avg_score": 0.72,
  "results": [
    {
      "id": "gsm8k_test_0",
      "question": "...",
      "prediction": "...",
      "answer": "18",
      "correct": true,
      "score": 1.0,
      "details": "pred_num=18.0, true_num=18.0"
    }
  ],
  "cost": {"total_tokens": 12345, "n_calls": 100}
}
```

## API Configuration

Routing rule:
- `--model_name gpt-4-o` uses the internal ByteDance Azure endpoint
- all other model names use zhizengzeng

Defaults can still be configured in `configs/eval_config.json`:

```json
{
  "llm": {
    "api_key": "...",
    "api_version": "2024-02-01",
    "azure_endpoint": "https://aidp.bytedance.net/api/modelhub/online/v2/crawl",
    "model": "gpt-4o-2024-11-20"
  }
}
```

If the endpoint is unreachable, use `--mock` for testing.

Zhizengzeng model list and pricing snapshot:
- [ZHIZENG_MODEL.md](/Users/bytedance/Documents/code/SkillsPolisher/ZHIZENG_MODEL.md)

## Dataset Registry

| Name | Evaluator | Size | Location |
|------|-----------|------|----------|
| `aqua_rat` | ExactMatch (MCQ) | 254 dev | CASE/data/AQUA_RAT |
| `gsm8k` | ExactMatch (numeric) | 1319 test | CASE/data/GSM8K |
| `tabmwp` | ExactMatch (numeric/MCQ) | 7686 test | CASE/data/tabmwp |
| `finqa` | ExactMatch (numeric) | ~8600 test | CASE/data/FinQA |
| `strategyqa` | ExactMatch (yes/no) | ~490 test | CASE/data/Strategyqa |
| `bpo_test` | LLM Judge | 200 | BPO/data/testset |
| `dolly_eval` | LLM Judge | 200 | BPO/data/testset |
| `self_instruct_eval` | LLM Judge | 252 | BPO/data/testset |
| `skillsbench` | LLM Judge | 89 | skillsbench/tasks |
| `demo_bank` | ExactMatch | 88 | splice/data |

## Architecture

```
eval_pipeline/
├── llm_client.py           # Azure OpenAI wrapper with retry + cost tracking
├── run_eval.py             # Main eval script
├── run_compare.py          # Multi-baseline comparison
├── configs/
│   └── eval_config.json    # API and path configuration
├── datasets/
│   ├── base.py             # EvalDataset + EvalSample abstractions
│   ├── aqua_rat.py         # AQUA-RAT loader
│   ├── gsm8k.py            # GSM8K loader
│   ├── tabmwp.py           # TabMWP loader
│   ├── finqa.py            # FinQA loader
│   ├── strategyqa.py       # StrategyQA loader
│   ├── bpo.py              # BPO test sets
│   ├── skillsbench.py      # SkillsBench tasks
│   ├── demo_bank.py        # Demo Bank
│   └── __init__.py         # DATASET_REGISTRY + load_dataset()
├── baselines/
│   ├── base.py             # Baseline abstract class
│   ├── zero_shot.py        # ZeroShotBaseline
│   ├── random_kshot.py     # RandomKShotBaseline
│   ├── case_bandit.py      # CASEBanditBaseline
│   ├── bpo_rewrite.py      # BPORewriteBaseline
│   └── __init__.py         # BASELINE_REGISTRY
├── evaluators/
│   ├── base.py             # Evaluator abstract class
│   ├── exact_match.py      # ExactMatchEvaluator
│   ├── llm_eval.py         # LLMEvaluator (judge)
│   └── __init__.py         # get_evaluator()
└── results/                # Project-level output root: ./results/{model_name}/{dataset_name}/
```

Current `run_compare.py` output location:

```text
./results/{model_name}/{dataset_name}/compare_result.json
```

It also writes per-method files such as:

```text
./results/{model_name}/{dataset_name}/zero_shot_result.json
./results/{model_name}/{dataset_name}/random_kshot_result.json
./results/{model_name}/{dataset_name}/case_bandit_result.json
./results/{model_name}/{dataset_name}/bpo_rewrite_result.json
```

If the experiment has not been run before, that directory is created automatically. Legacy flat files under `./results/{model_name}/` are migrated into dataset directories on startup.

Current `run_eval.py` output location:

```text
./results/{model_name}/{dataset_name}/{baseline}_result.json
```
