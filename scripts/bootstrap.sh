#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install -U pip
if [[ -n "${MELOTTS_ENGLISH_BUNDLE_SRC:-}" || "${MELOTTS_ENGLISH_ENABLED:-}" == "true" || "${MELOTTS_JAPANESE_ENABLED:-}" == "true" ]]; then
  extras="dev"
  if [[ -n "${MELOTTS_ENGLISH_BUNDLE_SRC:-}" || "${MELOTTS_ENGLISH_ENABLED:-}" == "true" ]]; then
    extras="${extras},english"
  fi
  if [[ "${MELOTTS_JAPANESE_ENABLED:-}" == "true" ]]; then
    extras="${extras},japanese"
  fi
  python -m pip install -e ".[${extras}]"
else
  python -m pip install -e ".[dev]"
fi
if [[ -n "${RKNN_LITE_WHEEL:-}" ]]; then
  python -m pip install "${RKNN_LITE_WHEEL}"
fi
if [[ -n "${MELOTTS_ENGLISH_BUNDLE_SRC:-}" ]]; then
  melotts-cli bootstrap --english-bundle-src "${MELOTTS_ENGLISH_BUNDLE_SRC}"
else
  melotts-cli bootstrap
fi
