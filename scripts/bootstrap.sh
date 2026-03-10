#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
if [[ -n "${RKNN_LITE_WHEEL:-}" ]]; then
  python -m pip install "${RKNN_LITE_WHEEL}"
fi
melotts-cli bootstrap
