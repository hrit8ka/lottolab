#!/usr/bin/env bash
# One-time setup on macOS. Run from the lottolab/ folder: bash scripts/setup_mac.sh
set -e
if ! command -v python3 >/dev/null; then echo "Install Python 3.11+ first (brew install python@3.11)"; exit 1; fi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
git init 2>/dev/null || true
echo "Done. Activate with: source .venv/bin/activate"
