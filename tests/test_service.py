from __future__ import annotations

import numpy as np

from melotts_server.config import Settings
from melotts_server.service import TTSService


class FakeEngine:
    loaded = True

    def warmup(self):
        return None

    def unload(self):
        return None

    def synthesize(self, text: str, speed: float, volume: float):
        return np.array([0.0, 0.25], dtype=np.float32)

    def synthesize_stream(self, text: str, speed: float, volume: float):
        yield type("Chunk", (), {"audio": np.array([0.0, 0.1], dtype=np.float32)})()


class FakePlayer:
    def __init__(self):
        self._state = type("State", (), {"current_request_id": None, "state": "idle", "queue_depth": 0, "detail": None})()
        self.device = "default"

    def available(self):
        return True

    def list_devices(self):
        return ["default"]

    @property
    def state(self):
        return self._state

    def stop(self):
        self._state.state = "stopped"

    def play_stream(self, request_id, chunks, device=None):
        self._state.current_request_id = request_id
        self._state.state = "playing"


def test_synthesize_returns_wav_bytes():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()
    result = service.synthesize("hello", 1.0, 1.0)
    assert result[:4] == b"RIFF"


def test_status_ready_when_engine_and_player_available():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()
    status = service.status()
    assert status["ready"] is True
