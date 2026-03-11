from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


ZH_REQUIRED_ARTIFACTS = (
    "encoder.onnx",
    "decoder.rknn",
    "g.bin",
    "lexicon.txt",
    "tokens.txt",
    "utils.py",
    "text",
)

EN_REQUIRED_ARTIFACTS = (
    "model/config.json",
    "export/encoder_cpu.onnx",
    "rknn/decoder_rk3588_fp16.rknn",
    "frontend/bert-base-uncased/config.json",
    "frontend/bert-base-uncased/model.safetensors",
    "frontend/bert-base-uncased/tokenizer.json",
    "frontend/bert-base-uncased/tokenizer_config.json",
    "frontend/bert-base-uncased/vocab.txt",
)


class ArtifactError(RuntimeError):
    pass


def _validate_required(model_dir: Path, required: Iterable[str], label: str) -> None:
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise ArtifactError(f"missing {label} artifacts in {model_dir}: {', '.join(missing)}")


def validate_artifacts(model_dir: Path) -> None:
    """Validate the legacy Chinese MeloTTS RKNN bundle."""
    _validate_required(model_dir, ZH_REQUIRED_ARTIFACTS, "Chinese")


def validate_english_bundle(bundle_dir: Path) -> None:
    """Validate the converted English-v3 hybrid bundle."""
    _validate_required(bundle_dir, EN_REQUIRED_ARTIFACTS, "English")


def bootstrap_artifacts(repo_id: str, model_dir: Path, allow_patterns: Iterable[str] | None = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns or ["*"]),
    )
    validate_artifacts(model_dir)
    return model_dir


def install_english_bundle(source_dir: Path, target_dir: Path) -> Path:
    """Copy a prepared English-v3 bundle into the server model directory."""
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    if not source_dir.exists():
        raise ArtifactError(f"english bundle source does not exist: {source_dir}")
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    validate_english_bundle(target_dir)
    return target_dir
