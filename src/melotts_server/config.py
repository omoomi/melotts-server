from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8123
    sample_rate: int = 44100
    model_repo_id: str = "happyme531/MeloTTS-RKNN2"
    model_dir: Path = Path("models/melotts-rknn2")
    runtime_lib_dir: Path = Path("lib")
    alsa_device: str = "auto"
    prebuffer_ms: int = 200
    max_text_chars: int = 1000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("MELOTTS_HOST", "127.0.0.1"),
            port=int(os.getenv("MELOTTS_PORT", "8123")),
            sample_rate=int(os.getenv("MELOTTS_SAMPLE_RATE", "44100")),
            model_repo_id=os.getenv("MELOTTS_MODEL_REPO_ID", "happyme531/MeloTTS-RKNN2"),
            model_dir=Path(os.getenv("MELOTTS_MODEL_DIR", "models/melotts-rknn2")),
            runtime_lib_dir=Path(os.getenv("MELOTTS_RUNTIME_LIB_DIR", "lib")),
            alsa_device=os.getenv("MELOTTS_ALSA_DEVICE", "auto"),
            prebuffer_ms=int(os.getenv("MELOTTS_PREBUFFER_MS", "200")),
            max_text_chars=int(os.getenv("MELOTTS_MAX_TEXT_CHARS", "1000")),
        )
