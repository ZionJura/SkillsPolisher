"""
download_datasets.py — Download all datasets into eval_pipeline/datasets/data/<name>/

Usage:
    python scripts/download_datasets.py                        # all datasets
    python scripts/download_datasets.py --all                  # same
    python scripts/download_datasets.py --dataset gsm8k        # single
    python scripts/download_datasets.py --dataset gsm8k,tabmwp # comma-separated
    python scripts/download_datasets.py --force                # re-download existing

Run from project root.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Project root / data directory ────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "eval_pipeline" / "datasets" / "data"

# Add project root to path so we can import data_utils
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Optional tqdm ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class _tqdm:
        """Minimal no-op tqdm replacement."""
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_data_dir(name: str) -> Path:
    d = DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _skip(path: Path, force: bool, label: str) -> bool:
    if path.exists() and not force:
        print(f"  [skip] {label} already exists (use --force to re-download)")
        return True
    return False


def _download_url(url: str, dest: Path, label: str = "") -> bool:
    """Download a URL to dest using requests with optional tqdm progress."""
    try:
        import requests
    except ImportError:
        print("  ERROR: 'requests' not installed. Run: pip install requests")
        return False

    label = label or dest.name
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            if HAS_TQDM and total:
                with _tqdm(total=total, unit="B", unit_scale=True, desc=f"  {label}") as bar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            else:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                print(f"  Downloaded {label}: {downloaded:,} bytes")
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── Per-dataset downloaders ───────────────────────────────────────────────────

def download_aqua_rat(data_dir: Path, force: bool) -> bool:
    """Download AQUA-RAT from HuggingFace datasets (deepmind/aqua_rat, config=raw)."""
    dev_file = data_dir / "aquarat_dev.csv"
    train_file = data_dir / "aquarat_train.csv"

    dev_exists = dev_file.exists()
    train_exists = train_file.exists()

    if dev_exists and train_exists and not force:
        print("  [skip] aquarat_dev.csv and aquarat_train.csv already exist")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' not installed. Run: pip install datasets")
        return False

    print("  Loading deepmind/aqua_rat (config=raw) from HuggingFace...")
    try:
        ds = load_dataset("deepmind/aqua_rat", "raw", trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return False

    fieldnames = ["question", "options", "rationale", "correct"]

    for split_name, csv_path in [("validation", dev_file), ("train", train_file)]:
        if csv_path.exists() and not force:
            print(f"  [skip] {csv_path.name} already exists")
            continue
        if split_name not in ds:
            print(f"  [warn] Split '{split_name}' not found, skipping {csv_path.name}")
            continue
        split_data = ds[split_name]
        print(f"  Writing {csv_path.name} ({len(split_data)} rows)...")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in split_data:
                writer.writerow({
                    "question": row.get("question", ""),
                    "options": str(row.get("options", [])),
                    "rationale": row.get("rationale", ""),
                    "correct": row.get("correct", ""),
                })
        print(f"  OK: {csv_path.name}")

    return dev_file.exists()


def download_gsm8k(data_dir: Path, force: bool) -> bool:
    """Download GSM8K from HuggingFace datasets (openai/gsm8k, config=main)."""
    test_file = data_dir / "gsm8k_test.jsonl"
    train_file = data_dir / "gsm8k_train.jsonl"

    if test_file.exists() and train_file.exists() and not force:
        print("  [skip] gsm8k_test.jsonl and gsm8k_train.jsonl already exist")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' not installed. Run: pip install datasets")
        return False

    print("  Loading openai/gsm8k (config=main) from HuggingFace...")
    try:
        ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return False

    for split_name, jsonl_path in [("test", test_file), ("train", train_file)]:
        if jsonl_path.exists() and not force:
            print(f"  [skip] {jsonl_path.name} already exists")
            continue
        if split_name not in ds:
            print(f"  [warn] Split '{split_name}' not found, skipping {jsonl_path.name}")
            continue
        split_data = ds[split_name]
        print(f"  Writing {jsonl_path.name} ({len(split_data)} rows)...")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in split_data:
                record = {"question": row["question"], "answer": row["answer"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  OK: {jsonl_path.name}")

    return test_file.exists()


def download_tabmwp(data_dir: Path, force: bool) -> bool:
    """Download TabMWP JSON files from GitHub (lupantech/PromptPG)."""
    base_url = "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/"
    files = ["problems_test.json", "problems_dev.json"]
    ok = True
    for fname in files:
        dest = data_dir / fname
        if _skip(dest, force, fname):
            continue
        print(f"  Downloading {fname}...")
        success = _download_url(base_url + fname, dest, fname)
        if not success:
            ok = False
    return ok


def download_finqa(data_dir: Path, force: bool) -> bool:
    """Download FinQA test.json from GitHub and convert to CSV."""
    csv_path = data_dir / "finqa_test.csv"
    if _skip(csv_path, force, "finqa_test.csv"):
        return True

    url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
    tmp_json = data_dir / "_finqa_test_raw.json"
    print("  Downloading FinQA test.json...")
    if not _download_url(url, tmp_json, "FinQA test.json"):
        return False

    print("  Converting FinQA test.json -> finqa_test.csv...")
    try:
        with open(tmp_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  ERROR reading downloaded JSON: {e}")
        return False

    fieldnames = ["question", "answer", "text", "table", "id", "program"]

    def _extract_text(pre_text, post_text):
        parts = []
        if isinstance(pre_text, list):
            parts.extend(str(s) for s in pre_text)
        elif pre_text:
            parts.append(str(pre_text))
        if isinstance(post_text, list):
            parts.extend(str(s) for s in post_text)
        elif post_text:
            parts.append(str(post_text))
        return " ".join(parts)

    def _extract_table(table_data):
        """Convert FinQA table (list of rows) to a readable string."""
        if not table_data:
            return ""
        if isinstance(table_data, str):
            return table_data
        if isinstance(table_data, list):
            rows = []
            for row in table_data:
                if isinstance(row, list):
                    rows.append(" | ".join(str(c) for c in row))
                else:
                    rows.append(str(row))
            return "\n".join(rows)
        return str(table_data)

    def _extract_program(ann):
        if not ann:
            return ""
        if isinstance(ann, dict):
            prog = ann.get("program", [])
            if isinstance(prog, list):
                return " ".join(str(s) for s in prog)
            return str(prog)
        return ""

    def _extract_answer(ann):
        if not ann:
            return ""
        if isinstance(ann, dict):
            return str(ann.get("exe_ans", ann.get("answer", "")))
        return str(ann)

    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in raw:
            qa = item.get("qa", {}) or {}
            pre_text = item.get("pre_text", [])
            post_text = item.get("post_text", [])
            table_data = item.get("table", [])
            writer.writerow({
                "question": qa.get("question", ""),
                "answer": _extract_answer(qa),
                "text": _extract_text(pre_text, post_text),
                "table": _extract_table(table_data),
                "id": item.get("id", ""),
                "program": _extract_program(qa),
            })
            rows_written += 1

    tmp_json.unlink(missing_ok=True)
    print(f"  OK: finqa_test.csv ({rows_written} rows)")
    return True


def download_strategyqa(data_dir: Path, force: bool) -> bool:
    """Download StrategyQA from HuggingFace datasets (wics/strategy-qa)."""
    test_file = data_dir / "strategyqa_test.csv"
    train_file = data_dir / "strategyqa_train.csv"

    if test_file.exists() and train_file.exists() and not force:
        print("  [skip] strategyqa_test.csv and strategyqa_train.csv already exist")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' not installed. Run: pip install datasets")
        return False

    print("  Loading wics/strategy-qa from HuggingFace...")
    try:
        ds = load_dataset("wics/strategy-qa", trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return False

    fieldnames = ["id", "question", "answer", "facts", "decomposition"]

    def _bool_to_yn(val):
        if isinstance(val, bool):
            return "yes" if val else "no"
        s = str(val).strip().lower()
        if s in ("true", "1", "yes"):
            return "yes"
        if s in ("false", "0", "no"):
            return "no"
        return s

    # wics/strategy-qa may only have 'test' split (the original labeled set)
    split_map = {}
    for split_name in ("test", "validation", "train"):
        if split_name in ds:
            split_map[split_name] = split_name

    # Map HF splits to output files
    output_map = [
        (["test", "validation"], test_file),
        (["train"], train_file),
    ]

    ok = False
    for candidate_splits, csv_path in output_map:
        if csv_path.exists() and not force:
            print(f"  [skip] {csv_path.name} already exists")
            ok = True
            continue
        chosen = None
        for s in candidate_splits:
            if s in ds:
                chosen = s
                break
        if chosen is None:
            print(f"  [warn] No suitable split found for {csv_path.name}, skipping")
            continue
        split_data = ds[chosen]
        print(f"  Writing {csv_path.name} (from HF split '{chosen}', {len(split_data)} rows)...")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in split_data:
                writer.writerow({
                    "id": row.get("qid", row.get("id", "")),
                    "question": row.get("question", ""),
                    "answer": _bool_to_yn(row.get("answer", "")),
                    "facts": repr(row.get("facts", [])),
                    "decomposition": repr(row.get("decomposition", row.get("steps", []))),
                })
        print(f"  OK: {csv_path.name}")
        ok = True

    return ok


def download_bpo(data_dir: Path, force: bool) -> bool:
    """Download BPO test files from THUDM/BPO GitHub."""
    base_url = "https://raw.githubusercontent.com/THUDM/BPO/main/data/testset/"
    files = ["bpo_test.json", "dolly_eval.json", "self_instruct_eval.json"]
    ok = True
    for fname in files:
        dest = data_dir / fname
        if _skip(dest, force, fname):
            continue
        print(f"  Downloading {fname}...")
        success = _download_url(base_url + fname, dest, fname)
        if not success:
            ok = False
    return ok


def download_skillsbench(data_dir: Path, force: bool) -> bool:
    """Clone benchflow-ai/skillsbench and copy tasks/ into data/skillsbench/."""
    # The loader expects a directory containing task subdirectories
    # We put the tasks/ contents directly into data/skillsbench/tasks/
    tasks_dir = data_dir / "tasks"

    if tasks_dir.exists() and any(tasks_dir.iterdir()) and not force:
        existing = sum(1 for _ in tasks_dir.iterdir() if _.is_dir())
        print(f"  [skip] data/skillsbench/tasks/ already has {existing} task dirs")
        return True

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clone_url = "https://github.com/benchflow-ai/skillsbench.git"
        print(f"  Cloning {clone_url} (shallow)...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", clone_url, str(tmp_path / "skillsbench")],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"  ERROR: git clone failed:\n{result.stderr}")
                return False
        except FileNotFoundError:
            print("  ERROR: git not found. Install git and retry.")
            return False
        except subprocess.TimeoutExpired:
            print("  ERROR: git clone timed out.")
            return False

        src_tasks = tmp_path / "skillsbench" / "tasks"
        if not src_tasks.exists():
            print(f"  ERROR: tasks/ not found in cloned repo")
            return False

        print(f"  Copying tasks/ to {tasks_dir}...")
        if tasks_dir.exists() and force:
            shutil.rmtree(tasks_dir)
        shutil.copytree(str(src_tasks), str(tasks_dir))

    n_tasks = sum(1 for _ in tasks_dir.iterdir() if _.is_dir())
    print(f"  OK: data/skillsbench/tasks/ ({n_tasks} tasks)")
    return True


def download_demo_bank(data_dir: Path, force: bool) -> bool:
    """Copy splice/data/demo_bank.json into data/demo_bank/demo_bank.json."""
    dest = data_dir / "demo_bank.json"
    if _skip(dest, force, "demo_bank.json"):
        return True

    src = PROJECT_ROOT / "splice" / "data" / "demo_bank.json"
    if not src.exists():
        print(f"  ERROR: source not found: {src}")
        return False

    shutil.copy2(str(src), str(dest))
    print(f"  OK: demo_bank.json (copied from splice/data/)")
    return True


# ── Registry ──────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "aqua_rat":    download_aqua_rat,
    "gsm8k":       download_gsm8k,
    "tabmwp":      download_tabmwp,
    "finqa":       download_finqa,
    "strategyqa":  download_strategyqa,
    "bpo":         download_bpo,
    "skillsbench": download_skillsbench,
    "demo_bank":   download_demo_bank,
}

ALL_DATASETS = list(DOWNLOADERS.keys())


def download_dataset(name: str, force: bool = False) -> bool:
    """Download a single dataset by name. Returns True on success."""
    if name not in DOWNLOADERS:
        print(f"  ERROR: unknown dataset '{name}'. Known: {', '.join(ALL_DATASETS)}")
        return False
    data_dir = _get_data_dir(name)
    print(f"\n[{name}] -> {data_dir}")
    try:
        return DOWNLOADERS[name](data_dir, force)
    except Exception as e:
        print(f"  ERROR: unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download all datasets for SkillsPolisher eval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py                          # all datasets
  python scripts/download_datasets.py --dataset gsm8k         # single dataset
  python scripts/download_datasets.py --dataset gsm8k,tabmwp  # multiple
  python scripts/download_datasets.py --all --force           # re-download all
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Comma-separated dataset name(s). If omitted, downloads all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets (same as omitting --dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (default: eval_pipeline/datasets/data/)",
    )
    args = parser.parse_args()

    # Override data dir if specified
    if args.data_dir:
        global DATA_DIR
        DATA_DIR = Path(args.data_dir).resolve()
        print(f"Data directory: {DATA_DIR}")

    # Determine which datasets to download
    if args.dataset:
        names = [n.strip() for n in args.dataset.split(",") if n.strip()]
        unknown = [n for n in names if n not in DOWNLOADERS]
        if unknown:
            print(f"ERROR: unknown datasets: {', '.join(unknown)}")
            print(f"Known: {', '.join(ALL_DATASETS)}")
            sys.exit(1)
    else:
        names = ALL_DATASETS

    print(f"=== SkillsPolisher Dataset Downloader ===")
    print(f"Datasets to download: {', '.join(names)}")
    print(f"Data directory: {DATA_DIR}")
    if args.force:
        print("Force mode: re-downloading existing files")

    results = {}
    for name in names:
        ok = download_dataset(name, force=args.force)
        results[name] = "OK" if ok else "FAILED"

    # Summary
    print(f"\n{'='*45}")
    print("SUMMARY")
    print(f"{'='*45}")
    success = [n for n, s in results.items() if s == "OK"]
    failed = [n for n, s in results.items() if s != "OK"]

    for name in names:
        status = results[name]
        mark = "+" if status == "OK" else "X"
        print(f"  [{mark}] {name}: {status}")

    print(f"\n{len(success)}/{len(names)} datasets ready")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All datasets downloaded successfully.")
        print(f"\nNext step:  python scripts/verify_setup.py")


if __name__ == "__main__":
    main()
