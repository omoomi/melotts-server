from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MODEL_ROOT = Path("models")
DEFAULT_CHINESE_MODEL_DIR = MODEL_ROOT / "chinese"
DEFAULT_ENGLISH_BUNDLE_DIR = MODEL_ROOT / "english-v3"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8123
    sample_rate: int = 44100
    model_repo_id: str = "happyme531/MeloTTS-RKNN2"
    model_dir: Path = DEFAULT_CHINESE_MODEL_DIR
    runtime_lib_dir: Path = Path("lib")
    alsa_device: str = "auto"
    prebuffer_ms: int = 200
    max_text_chars: int = 1000
    english_enabled: bool = False
    english_bundle_dir: Path = DEFAULT_ENGLISH_BUNDLE_DIR
    english_latent_frames: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        english_bundle_dir = Path(os.getenv("MELOTTS_ENGLISH_BUNDLE_DIR", str(DEFAULT_ENGLISH_BUNDLE_DIR)))
        english_enabled_default = english_bundle_dir.exists()
        return cls(
            host=os.getenv("MELOTTS_HOST", "127.0.0.1"),
            port=int(os.getenv("MELOTTS_PORT", "8123")),
            sample_rate=int(os.getenv("MELOTTS_SAMPLE_RATE", "44100")),
            model_repo_id=os.getenv("MELOTTS_MODEL_REPO_ID", "happyme531/MeloTTS-RKNN2"),
            model_dir=Path(os.getenv("MELOTTS_MODEL_DIR", str(DEFAULT_CHINESE_MODEL_DIR))),
            runtime_lib_dir=Path(os.getenv("MELOTTS_RUNTIME_LIB_DIR", "lib")),
            alsa_device=os.getenv("MELOTTS_ALSA_DEVICE", "auto"),
            prebuffer_ms=int(os.getenv("MELOTTS_PREBUFFER_MS", "200")),
            max_text_chars=int(os.getenv("MELOTTS_MAX_TEXT_CHARS", "1000")),
            english_enabled=_env_bool("MELOTTS_ENGLISH_ENABLED", english_enabled_default),
            english_bundle_dir=english_bundle_dir,
            english_latent_frames=int(os.getenv("MELOTTS_ENGLISH_LATENT_FRAMES", "256")),
        )
