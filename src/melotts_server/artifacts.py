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

JA_REQUIRED_ARTIFACTS = (
    "model/config.json",
    "export/encoder_cpu.onnx",
    "rknn/decoder_ja_rk3588_fp16.rknn",
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


def validate_japanese_bundle(bundle_dir: Path) -> None:
    """Validate the converted Japanese hybrid bundle."""
    _validate_required(bundle_dir, JA_REQUIRED_ARTIFACTS, "Japanese")


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


def install_japanese_bundle(source_root: Path, target_dir: Path) -> Path:
    """Copy a prepared Japanese hybrid bundle from a split source tree."""
    source_root = source_root.resolve()
    target_dir = target_dir.resolve()
    if not source_root.exists():
        raise ArtifactError(f"japanese source root does not exist: {source_root}")

    copies = (
        ("bundle_ja/model/config.json", "model/config.json"),
        ("bundle_ja/model/checkpoint.pth", "model/checkpoint.pth"),
        ("bundle_ja/manifest.json", "manifest.json"),
        ("export_ja/encoder_cpu.onnx", "export/encoder_cpu.onnx"),
        ("export_ja/decoder_npu.onnx", "export/decoder_npu.onnx"),
        ("decoder_ja_rk3588_fp16.rknn", "rknn/decoder_ja_rk3588_fp16.rknn"),
        ("decoder_ja_rk3588_fp16.rknn.json", "rknn/decoder_ja_rk3588_fp16.rknn.json"),
    )
    for source_rel, target_rel in copies:
        source_path = source_root / source_rel
        if not source_path.exists():
            continue
        target_path = target_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    validate_japanese_bundle(target_dir)
    return target_dir
