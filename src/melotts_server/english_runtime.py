from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort

from .artifacts import validate_english_bundle
from .config import Settings
from .english_frontend import MeloEnglishFrontend

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EnglishSynthesisChunk:
    index: int
    text: str
    audio: np.ndarray


class EnglishOnnxEncoder:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.vocab_size = 4096
        self.tone_size = 16
        self.language_size = 8
        self._load_embedding_sizes(model_path)

    def _load_embedding_sizes(self, model_path: Path) -> None:
        import onnx

        model = onnx.load(str(model_path))
        for init in model.graph.initializer:
            name = init.name
            if name.endswith("enc_p.emb.weight") and init.dims:
                self.vocab_size = int(init.dims[0])
            elif name.endswith("enc_p.tone_emb.weight") and init.dims:
                self.tone_size = int(init.dims[0])
            elif name.endswith("enc_p.language_emb.weight") and init.dims:
                self.language_size = int(init.dims[0])

    def infer(
        self,
        token_ids: np.ndarray,
        tone_ids: np.ndarray,
        lang_ids: np.ndarray,
        bert: np.ndarray,
        ja_bert: np.ndarray,
        speaker_id: int,
        length_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t = len(token_ids)
        feed: dict[str, np.ndarray] = {
            "x": token_ids.astype(np.int64)[None, :],
            "x_lengths": np.asarray([t], dtype=np.int64),
            "sid": np.asarray([speaker_id], dtype=np.int64),
            "tone": tone_ids.astype(np.int64)[None, :],
            "language": lang_ids.astype(np.int64)[None, :],
            "bert": bert[None, :, :].astype(np.float32),
            "ja_bert": ja_bert[None, :, :].astype(np.float32),
            "length_scale": np.asarray([[[length_scale]]], dtype=np.float32),
            "noise_scale": np.asarray([[[0.667]]], dtype=np.float32),
            "sdp_ratio": np.asarray([[[0.2]]], dtype=np.float32),
            "noise_scale_w": np.asarray([[[0.8]]], dtype=np.float32),
        }
        out = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})
        if len(out) < 2:
            raise RuntimeError("English encoder ONNX must output (z_masked, g)")
        return np.asarray(out[0], dtype=np.float32), np.asarray(out[1], dtype=np.float32)


class EnglishRknnDecoder:
    def __init__(self, model_path: Path):
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "rknn-toolkit-lite2 is required for the English-v3 runtime."
            ) from exc
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNLite init_runtime failed: {ret}")

    def infer(self, z_masked: np.ndarray, g: np.ndarray) -> np.ndarray:
        out = self.rknn.inference(inputs=[z_masked, g])
        return np.asarray(out[0]).squeeze()

    def release(self):
        self.rknn.release()


class MeloTTSEnglishV3Engine:
    """English-v3 hybrid runtime using CPU ONNX encoder and RKNN decoder."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaded = False
        self.frontend: MeloEnglishFrontend | None = None
        self.encoder: EnglishOnnxEncoder | None = None
        self.decoder: EnglishRknnDecoder | None = None
        self._speaker_id = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        bundle = self._settings.english_bundle_dir
        validate_english_bundle(bundle)
        self.frontend = MeloEnglishFrontend(
            config_path=bundle / "model" / "config.json",
            frontend_assets=bundle / "frontend" / "bert-base-uncased",
            use_bert=True,
        )
        if self.frontend.valid_speaker_ids:
            self._speaker_id = self.frontend.valid_speaker_ids[0]
        self.encoder = EnglishOnnxEncoder(bundle / "export" / "encoder_cpu.onnx")
        self.decoder = EnglishRknnDecoder(bundle / "rknn" / "decoder_rk3588_fp16.rknn")
        self._loaded = True

    def warmup(self) -> None:
        self.load()
        list(self.synthesize_stream("english warmup", speed=1.0))

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

    def synthesize_stream(self, text: str, speed: float = 1.0, volume: float = 1.0) -> Iterator[EnglishSynthesisChunk]:
        self.load()
        assert self.frontend is not None
        assert self.encoder is not None
        assert self.decoder is not None

        for index, sentence in enumerate(_split_sentences_en(text)):
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
                "English sentence %d text=%r token_count=%d latent_frames=%d",
                index,
                sentence[:120],
                len(features.token_ids),
                z_masked.shape[-1],
            )
            audio = self._decode_chunked(z_masked, g)
            if volume != 1.0:
                audio = np.clip(audio * volume, -1.0, 1.0)
            pause = np.zeros((int((self._settings.sample_rate * 0.05) / max(speed, 0.1)),), dtype=np.float32)
            yield EnglishSynthesisChunk(index=index, text=sentence, audio=np.concatenate((audio, pause), axis=-1))

    def _decode_chunked(self, z_masked: np.ndarray, g: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        t = z_masked.shape[-1]
        frame = self._settings.english_latent_frames
        chunks: list[np.ndarray] = []
        for start in range(0, t, frame):
            end = min(start + frame, t)
            sub = z_masked[:, :, start:end]
            valid_latent = sub.shape[-1]
            if valid_latent < frame:
                pad = np.zeros((sub.shape[0], sub.shape[1], frame - valid_latent), dtype=np.float32)
                sub = np.concatenate([sub, pad], axis=-1)
            wav = self.decoder.infer(sub, g)
            chunk_audio = np.asarray(wav[: valid_latent * 512], dtype=np.float32)
            logger.info(
                "English decode chunk start=%d end=%d valid_latent=%d samples=%d",
                start,
                end,
                valid_latent,
                len(chunk_audio),
            )
            chunks.append(chunk_audio)
        if not chunks:
            return np.zeros(1, dtype=np.float32)
        wav = np.concatenate(chunks, axis=-1)
        max_abs = np.max(np.abs(wav)) if wav.size else 1.0
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav.astype(np.float32)


def _split_sentences_en(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]
