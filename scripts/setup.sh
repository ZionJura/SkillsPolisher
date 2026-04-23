#!/bin/bash
# Setup script for SkillsPolisher on a new machine
set -e

echo "=== SkillsPolisher Setup ==="
echo "Python: $(python3 --version)"
echo "Working directory: $(pwd)"

# Ensure we are in the project root (directory containing eval_pipeline/)
if [ ! -d "eval_pipeline" ]; then
    echo "ERROR: Run this script from the project root (the directory containing eval_pipeline/)"
    echo "  cd SkillsPolisher && bash scripts/setup.sh"
    exit 1
fi

echo ""
echo "--- Installing core dependencies ---"
pip install openai datasets requests tqdm

echo ""
echo "--- Optional: TOML support for Python < 3.11 ---"
python3 -c "import tomllib" 2>/dev/null || pip install tomli || true

echo ""
echo "--- Verifying imports ---"
python3 -c "import openai; print('  openai:', openai.__version__)"
python3 -c "import datasets; print('  datasets:', datasets.__version__)"
python3 -c "import requests; print('  requests:', requests.__version__)"
python3 -c "import tqdm; print('  tqdm:', tqdm.__version__)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  python scripts/download_datasets.py        # download all datasets (~500MB)"
echo "  python scripts/verify_setup.py             # verify everything works"
echo ""
echo "Optional (GPU required, for SPLICE rewriter):"
echo "  pip install torch transformers bitsandbytes"
