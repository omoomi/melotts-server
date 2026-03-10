from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .artifacts import validate_artifacts
from .config import Settings
from .external_model import load_utils_module


@dataclass(slots=True)
class SynthesisChunk:
    index: int
    text: str
    audio: np.ndarray


class MeloTTSRKNNEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self._session_enc = None
        self._decoder = None
        self._g = None
        self._dec_len = 0
        self._modules = None
        self._lexicon = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            validate_artifacts(self._settings.model_dir)
            self._load_runtime_libs(self._settings.runtime_lib_dir)

            import onnxruntime as ort
            try:
                from rknnlite.api import RKNNLite
            except ImportError as exc:
                raise RuntimeError(
                    "rknnlite is not installed. Install the Rockchip rknn-toolkit-lite2 wheel in the active virtualenv."
                ) from exc

            self._modules = load_utils_module(self._settings.model_dir)
            self._session_enc = ort.InferenceSession(
                str(self._settings.model_dir / "encoder.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=ort.SessionOptions(),
            )
            self._decoder = RKNNLite()
            ret = self._decoder.load_rknn(str(self._settings.model_dir / "decoder.rknn"))
            if ret != 0:
                raise RuntimeError(f"load_rknn failed with code {ret}")
            ret = self._decoder.init_runtime()
            if ret != 0:
                raise RuntimeError(f"init_runtime failed with code {ret}")

            self._g = np.fromfile(self._settings.model_dir / "g.bin", dtype=np.float32).reshape(1, 256, 1)
            self._lexicon = self._modules.utils.Lexicon(
                str(self._settings.model_dir / "lexicon.txt"),
                str(self._settings.model_dir / "tokens.txt"),
            )
            self._dec_len = 65536 // 512
            self._loaded = True

    def warmup(self) -> None:
        self.load()
        list(self.synthesize_stream("你好", speed=1.0))

    def unload(self) -> None:
        with self._lock:
            if self._decoder is not None:
                self._decoder.release()
            self._decoder = None
            self._session_enc = None
            self._loaded = False

    def synthesize(self, text: str, speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
        chunks = [chunk.audio for chunk in self.synthesize_stream(text, speed=speed, volume=volume)]
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=-1)

    def synthesize_stream(self, text: str, speed: float = 1.0, volume: float = 1.0) -> Iterator[SynthesisChunk]:
        self.load()
        assert self._modules is not None
        assert self._session_enc is not None
        assert self._decoder is not None
        assert self._g is not None
        assert self._lexicon is not None

        split_sentences = self._modules.utils.split_sentences_zh(text)
        for index, sentence in enumerate(split_sentences):
            if not sentence.strip():
                continue
            audio = self._synthesize_sentence(sentence, speed)
            if volume != 1.0:
                audio = np.clip(audio * volume, -1.0, 1.0)
            yield SynthesisChunk(index=index, text=sentence, audio=audio.astype(np.float32))

    def _load_runtime_libs(self, runtime_lib_dir: Path) -> None:
        runtime_lib_dir = runtime_lib_dir.resolve()
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if str(runtime_lib_dir) not in current.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{runtime_lib_dir}:{current}" if current else str(runtime_lib_dir)
        for lib_name in ("librknnrt.so",):
            lib_path = runtime_lib_dir / lib_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path))

    def _synthesize_sentence(self, sentence: str, speed: float) -> np.ndarray:
        utils = self._modules.utils
        phone_str, yinjie_num, phones, tones = self._lexicon.convert(sentence)
        phone_str = utils.intersperse(phone_str, 0)
        phones = np.array(utils.intersperse(phones, 0), dtype=np.int32)
        tones = np.array(utils.intersperse(tones, 0), dtype=np.int32)
        yinjie_num = np.array(yinjie_num, dtype=np.int32) * 2
        yinjie_num[0] += 1

        pron_slices = self._generate_pronounce_slice(yinjie_num)
        phone_len = phones.shape[-1]
        language = np.array([3] * phone_len, dtype=np.int32)

        z_p, pronoun_lens, audio_len = self._session_enc.run(
            None,
            input_feed={
                "phone": phones,
                "g": self._g,
                "tone": tones,
                "language": language,
                "noise_scale": np.array([0], dtype=np.float32),
                "length_scale": np.array([1.0 / speed], dtype=np.float32),
                "noise_scale_w": np.array([0], dtype=np.float32),
                "sdp_ratio": np.array([0], dtype=np.float32),
            },
        )

        audio_len = int(audio_len[0])
        actual_size = z_p.shape[-1]
        dec_slice_num = int(np.ceil(actual_size / self._dec_len))
        z_p = np.pad(
            z_p,
            pad_width=((0, 0), (0, 0), (0, dec_slice_num * self._dec_len - actual_size)),
            mode="constant",
            constant_values=0,
        )

        pron_num = self._generate_word_pron_num(pronoun_lens, pron_slices)
        sub_audio_list: list[np.ndarray] = []
        pron_num_slices, zp_slices, strip_flags, _, is_long = self._generate_decode_slices(pron_num, self._dec_len)
        for i, (pron_slice, zp_slice) in enumerate(zip(pron_num_slices, zp_slices)):
            pron_start, pron_end = pron_slice
            zp_start, zp_end = zp_slice
            if is_long[i]:
                sub_audio_list.extend(self._decode_long_word(z_p[..., zp_start:zp_end], self._g, self._dec_len))
                continue

            sub_dec_len = zp_end - zp_start
            sub_audio_len = 512 * sub_dec_len
            zp_value = z_p[..., zp_start:zp_end]
            if zp_value.shape[-1] < self._dec_len:
                zp_value = np.concatenate(
                    (zp_value, np.zeros((*zp_value.shape[:-1], self._dec_len - zp_value.shape[-1]), dtype=np.float32)),
                    axis=-1,
                )
            audio = self._decoder.inference(inputs=[zp_value, self._g])[0].flatten()
            audio = audio[:sub_audio_len]
            if strip_flags[i][0]:
                head = 512 * pron_num[pron_start]
                audio = audio[head:]
            if strip_flags[i][1]:
                tail = 512 * pron_num[pron_end - 1]
                audio = audio[:-tail]
            sub_audio_list.append(audio)

        sub_audio = self._merge_sub_audio(sub_audio_list, 0, audio_len)
        pause = np.zeros((int((self._settings.sample_rate * 0.05) / speed),), dtype=np.float32)
        return np.concatenate((sub_audio.astype(np.float32), pause), axis=-1)

    def _decode_long_word(self, z_p: np.ndarray, g: np.ndarray, dec_len: int) -> Iterable[np.ndarray]:
        z_p_len = z_p.shape[-1]
        slice_num = int(np.ceil(z_p_len / dec_len))
        sub_audio_list = []
        for i in range(slice_num):
            z_p_slice = z_p[..., i * dec_len : (i + 1) * dec_len]
            sub_dec_len = z_p_slice.shape[-1]
            sub_audio_len = 512 * sub_dec_len
            if z_p_slice.shape[-1] < dec_len:
                z_p_slice = np.concatenate(
                    (z_p_slice, np.zeros((*z_p_slice.shape[:-1], dec_len - z_p_slice.shape[-1]), dtype=np.float32)),
                    axis=-1,
                )
            audio = self._decoder.inference(inputs=[z_p_slice, g])[0].flatten()
            sub_audio_list.append(audio[:sub_audio_len])
        return sub_audio_list

    @staticmethod
    def _merge_sub_audio(sub_audio_list: list[np.ndarray], pad_size: int, audio_len: int) -> np.ndarray:
        if pad_size > 0:
            for i in range(len(sub_audio_list) - 1):
                sub_audio_list[i][-pad_size:] += sub_audio_list[i + 1][:pad_size]
                sub_audio_list[i][-pad_size:] /= 2
                if i > 0:
                    sub_audio_list[i] = sub_audio_list[i][pad_size:]
        sub_audio = np.concatenate(sub_audio_list, axis=-1)
        return sub_audio[:audio_len]

    @staticmethod
    def _generate_pronounce_slice(yinjie_num: np.ndarray) -> list[slice]:
        pron_slice = []
        start = 0
        for count in yinjie_num:
            end = start + int(count)
            pron_slice.append(slice(start, end))
            start = end
        return pron_slice

    @staticmethod
    def _generate_word_pron_num(pron_lens: np.ndarray, pron_slices: list[slice]) -> list[int]:
        return [int(pron_lens[item].sum()) for item in pron_slices]

    @staticmethod
    def _generate_decode_slices(pron_num: list[int], dec_len: int):
        pron_slices = []
        zp_slices = []
        strip_flags = []
        pron_lens = []
        is_long = []
        start = end = 0
        zp_start = zp_end = 0
        while end < len(pron_num):
            if len(pron_lens) > 0 and pron_lens[-1] > 2 and pron_num[end] < dec_len:
                prev_end = pron_slices[-1][1]
                pad_size = pron_num[prev_end - 1] + pron_num[prev_end - 2]
                start = end - 2
                zp_start = zp_end - pad_size
                strip_head, strip_tail = True, False
                if is_long[-1]:
                    strip_flags[-1][1] = False
                else:
                    strip_flags[-1][1] = True
            else:
                start = end
                zp_start = zp_end
                strip_head, strip_tail = False, False
                if len(strip_flags) > 0:
                    strip_flags[-1][1] = False
            sub_dec_len = abs(zp_end - zp_start)
            while end < len(pron_num) and sub_dec_len < dec_len:
                sub_dec_len += pron_num[end]
                end += 1
            if end - start == 1 and sub_dec_len > dec_len:
                is_long.append(True)
                if len(strip_flags) > 0:
                    strip_flags[-1][1] = False
            else:
                is_long.append(False)
            if sub_dec_len > dec_len:
                sub_dec_len -= pron_num[end - 1]
                end -= 1
            zp_end = zp_start + sub_dec_len
            pron_slices.append([start, end])
            zp_slices.append([zp_start, zp_end])
            strip_flags.append([strip_head, strip_tail])
            pron_lens.append(end - start)
        return pron_slices, zp_slices, strip_flags, pron_lens, is_long


def wav_bytes_from_float(audio: np.ndarray, sample_rate: int) -> bytes:
    import io
    import wave

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()
