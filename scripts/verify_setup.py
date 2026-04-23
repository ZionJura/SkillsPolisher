"""
verify_setup.py — Verify that SkillsPolisher is set up correctly.

Checks:
  1. Python version >= 3.9
  2. Required packages importable
  3. Each dataset can be loaded (n=1 sample)
  4. API connectivity (unless --no-api)

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --no-api    # skip API connectivity test
    python scripts/verify_setup.py --dataset gsm8k  # single dataset

Run from project root.
"""

import argparse
import sys
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Formatting helpers ────────────────────────────────────────────────────────

def _col(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    if len(text) > width:
        text = text[:width - 2] + ".."
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def _print_table(headers: list, rows: list, col_widths: list):
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(_col(h, w) for h, w in zip(headers, col_widths)) + " |"
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(_col(v, w) for v, w in zip(row, col_widths)) + " |"
        print(line)
    print(sep)


# ── Check 1: Python version ───────────────────────────────────────────────────

def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 9)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {major}.{minor} (need >= 3.9)")
    return ok


# ── Check 2: Package imports ──────────────────────────────────────────────────

REQUIRED_PACKAGES = ["openai", "requests"]
OPTIONAL_PACKAGES = ["datasets", "tqdm", "tomllib", "tomli"]


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"[OK  ] {pkg} {ver}")
        except ImportError:
            print(f"[FAIL] {pkg} not installed  --> pip install {pkg}")
            all_ok = False
    for pkg in OPTIONAL_PACKAGES:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"[OK  ] {pkg} {ver} (optional)")
        except ImportError:
            print(f"[warn] {pkg} not installed (optional)")
    return all_ok


# ── Check 3: Dataset loading ──────────────────────────────────────────────────

# (dataset_name, split, constructor_kwargs)
DATASET_CHECKS = [
    ("aqua_rat",          "dev",   {}),
    ("gsm8k",             "test",  {}),
    ("tabmwp",            "test",  {}),
    ("finqa",             "test",  {}),
    ("strategyqa",        "test",  {}),
    ("bpo_test",          "test",  {}),
    ("dolly_eval",        "test",  {}),
    ("self_instruct_eval","test",  {}),
    ("skillsbench",       "test",  {}),
    ("demo_bank",         "train", {}),
]


def check_datasets(only_dataset: str = None) -> list:
    """
    Try loading each dataset with n=1. Returns rows for the summary table.
    Row format: [dataset, status, n_samples, first_question]
    """
    try:
        from eval_pipeline.datasets import load_dataset, DATASET_REGISTRY
        from eval_pipeline.datasets.data_utils import DataNotFoundError
    except ImportError as e:
        print(f"ERROR: could not import eval_pipeline.datasets: {e}")
        return []

    rows = []
    checks = DATASET_CHECKS
    if only_dataset:
        checks = [(n, s, kw) for n, s, kw in checks if n == only_dataset]

    for ds_name, split, kwargs in checks:
        try:
            ds = load_dataset(ds_name, split=split, **kwargs)
            n = len(ds)
            if n > 0:
                first_q = ds[0].question[:60].replace("\n", " ")
                status = "OK"
            else:
                first_q = "(empty)"
                status = "EMPTY"
        except DataNotFoundError as e:
            n = 0
            first_q = "run: python scripts/download_datasets.py"
            status = "NOT_FOUND"
        except FileNotFoundError as e:
            n = 0
            first_q = str(e)[:60]
            status = "NOT_FOUND"
        except Exception as e:
            n = 0
            first_q = str(e)[:60]
            status = f"ERROR"

        rows.append([ds_name, status, str(n), first_q])

    return rows


# ── Check 4: API connectivity ─────────────────────────────────────────────────

def check_api() -> bool:
    """Call LLMClient with a minimal 1-token request to verify connectivity."""
    try:
        from eval_pipeline.llm_client import LLMClient
    except ImportError as e:
        print(f"[FAIL] Could not import LLMClient: {e}")
        return False

    print("  Testing API connectivity (model=gpt-4o-mini, 1 token)...")
    try:
        client = LLMClient(model="gpt-4o-mini")
        response = client.complete(
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
        )
        text = response.strip() if response else ""
        print(f"  API response: {repr(text[:50])}")
        print("[OK  ] API connectivity")
        return True
    except ConnectionError as e:
        print(f"[FAIL] API unreachable: {e}")
        print("       Use --mock in run_eval.py / run_compare.py for offline testing")
        return False
    except Exception as e:
        print(f"[FAIL] API error: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify SkillsPolisher setup",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip API connectivity check",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Check only this dataset (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SkillsPolisher Setup Verification")
    print("=" * 60)

    all_ok = True

    # 1. Python version
    print("\n--- Python version ---")
    if not check_python_version():
        all_ok = False

    # 2. Packages
    print("\n--- Package imports ---")
    if not check_packages():
        all_ok = False

    # 3. Datasets
    print("\n--- Dataset loading ---")
    ds_rows = check_datasets(only_dataset=args.dataset)
    if ds_rows:
        col_widths = [20, 10, 10, 62]
        headers = ["dataset", "status", "n_samples", "first_question[:60]"]
        _print_table(headers, ds_rows, col_widths)

        failed_ds = [r[0] for r in ds_rows if r[1] not in ("OK", "EMPTY")]
        if failed_ds:
            print(f"\n  Datasets not ready: {', '.join(failed_ds)}")
            print(f"  Run: python scripts/download_datasets.py --dataset {','.join(failed_ds)}")
    else:
        print("  No datasets checked.")

    # 4. API
    if not args.no_api:
        print("\n--- API connectivity ---")
        check_api()
    else:
        print("\n--- API connectivity: SKIPPED (--no-api) ---")

    # Summary
    print("\n" + "=" * 60)
    ds_ok_count = sum(1 for r in ds_rows if r[1] == "OK")
    ds_total = len(ds_rows)
    print(f"Datasets ready: {ds_ok_count}/{ds_total}")
    if all_ok and ds_ok_count == ds_total:
        print("Setup looks good. Ready to run evaluations.")
    else:
        print("Some checks failed or datasets are missing.")
        print("Run 'python scripts/download_datasets.py' to fix missing datasets.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
