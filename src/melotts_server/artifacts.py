from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


REQUIRED_ARTIFACTS = (
    "encoder.onnx",
    "decoder.rknn",
    "g.bin",
    "lexicon.txt",
    "tokens.txt",
    "utils.py",
    "text",
)


class ArtifactError(RuntimeError):
    pass


def validate_artifacts(model_dir: Path) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (model_dir / name).exists()]
    if missing:
        raise ArtifactError(f"missing model artifacts in {model_dir}: {', '.join(missing)}")


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

