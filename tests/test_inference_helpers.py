from __future__ import annotations

import numpy as np

from melotts_server.inference import MeloTTSRKNNEngine, wav_bytes_from_float


def test_generate_pronounce_slice():
    result = MeloTTSRKNNEngine._generate_pronounce_slice(np.array([2, 3], dtype=np.int32))
    assert result[0].start == 0
    assert result[0].stop == 2
    assert result[1].start == 2
    assert result[1].stop == 5


def test_wav_bytes_from_float_has_riff_header():
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    payload = wav_bytes_from_float(audio, 44100)
    assert payload[:4] == b"RIFF"

