from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .english_frontend import FrontendOutput


class MeloJapaneseFrontend:
    """Japanese frontend backed by the Melo text package shipped with the Chinese bundle."""

    def __init__(self, text_assets_root: Path):
        self._text_assets_root = text_assets_root.resolve()
        self._cleaner = None
        self._text_module = None

    def load(self) -> None:
        self._ensure_dependencies()
        if str(self._text_assets_root) not in sys.path:
            sys.path.insert(0, str(self._text_assets_root))
        self._cleaner = importlib.import_module("text.cleaner")
        self._text_module = importlib.import_module("text")

    def encode(self, text: str) -> FrontendOutput:
        if self._cleaner is None or self._text_module is None:
            self.load()
        assert self._cleaner is not None
        assert self._text_module is not None

        norm_text, phones, tones, _word2ph = self._cleaner.clean_text(text, "JP")
        _, phone_ids, tone_ids, lang_ids = self._text_module.cleaned_text_to_sequence(phones, tones, "JP")
        token_count = len(phone_ids)
        return FrontendOutput(
            token_ids=np.asarray(phone_ids, dtype=np.int64),
            tone_ids=np.asarray(tone_ids, dtype=np.int64),
            lang_ids=np.asarray(lang_ids, dtype=np.int64),
            bert=np.zeros((1024, token_count), dtype=np.float32),
            ja_bert=np.zeros((768, token_count), dtype=np.float32),
        )

    @staticmethod
    def _ensure_dependencies() -> None:
        missing = [
            module_name
            for module_name in ("MeCab", "num2words")
            if importlib.util.find_spec(module_name) is None
        ]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(
                f"Missing Japanese frontend deps: {names}. Install the 'japanese' extra for melotts-server."
            )
