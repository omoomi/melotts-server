from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .artifacts import validate_japanese_bundle
from .config import Settings
from .english_runtime import EnglishOnnxEncoder, EnglishRknnDecoder
from .japanese_frontend import MeloJapaneseFrontend

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JapaneseSynthesisChunk:
    index: int
    text: str
    audio: np.ndarray


class MeloTTSJapaneseEngine:
    """Japanese hybrid runtime using CPU ONNX encoder and RKNN decoder."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaded = False
        self.frontend: MeloJapaneseFrontend | None = None
        self.encoder: EnglishOnnxEncoder | None = None
        self.decoder: EnglishRknnDecoder | None = None
        self._speaker_id = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        bundle = self._settings.japanese_bundle_dir
        validate_japanese_bundle(bundle)
        self.frontend = MeloJapaneseFrontend(self._settings.model_dir)
        self.encoder = EnglishOnnxEncoder(bundle / "export" / "encoder_cpu.onnx")
        self.decoder = EnglishRknnDecoder(bundle / "rknn" / "decoder_ja_rk3588_fp16.rknn")
        self._loaded = True

    def warmup(self) -> None:
        self.load()
        list(self.synthesize_stream("これは日本語のウォームアップです。", speed=1.0))

    def unload(self) -> None:
        if self.decoder is not None:
            self.decoder.release()
        self.decoder = None
        self.encoder = None
        self.frontend = None
        self._loaded = False

    def synthesize(self, text: str, speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
        chunks = [chunk.audio for chunk in self.synthesize_stream(text, speed=speed, volume=volume)]
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=-1)

    def synthesize_stream(self, text: str, speed: float = 1.0, volume: float = 1.0) -> Iterator[JapaneseSynthesisChunk]:
        self.load()
        assert self.frontend is not None
        assert self.encoder is not None
        assert self.decoder is not None

        for index, sentence in enumerate(_split_sentences_ja(text)):
            if not sentence.strip():
                continue
            features = self.frontend.encode(sentence)
            z_masked, g = self.encoder.infer(
                features.token_ids,
                features.tone_ids,
                features.lang_ids,
                features.bert,
                features.ja_bert,
                self._speaker_id,
                length_scale=1.0 / max(speed, 0.1),
            )
            logger.info(
                "Japanese sentence %d text=%r token_count=%d latent_frames=%d",
                index,
                sentence[:120],
                len(features.token_ids),
                z_masked.shape[-1],
            )
            audio = self._decode_chunked(z_masked, g)
            if volume != 1.0:
                audio = np.clip(audio * volume, -1.0, 1.0)
            pause = np.zeros((int((self._settings.sample_rate * 0.05) / max(speed, 0.1)),), dtype=np.float32)
            yield JapaneseSynthesisChunk(index=index, text=sentence, audio=np.concatenate((audio, pause), axis=-1))

    def _decode_chunked(self, z_masked: np.ndarray, g: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        t = z_masked.shape[-1]
        frame = self._settings.japanese_latent_frames
        chunks: list[np.ndarray] = []
        for start in range(0, t, frame):
            end = min(start + frame, t)
            sub = z_masked[:, :, start:end]
            valid_latent = sub.shape[-1]
            if valid_latent < frame:
                pad = np.zeros((sub.shape[0], sub.shape[1], frame - valid_latent), dtype=np.float32)
                sub = np.concatenate([sub, pad], axis=-1)
            wav = self.decoder.infer(sub, g)
            chunks.append(np.asarray(wav[: valid_latent * 512], dtype=np.float32))
        if not chunks:
            return np.zeros(1, dtype=np.float32)
        wav = np.concatenate(chunks, axis=-1)
        max_abs = np.max(np.abs(wav)) if wav.size else 1.0
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav.astype(np.float32)


def _split_sentences_ja(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]
