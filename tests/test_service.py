from __future__ import annotations

import time
from unittest.mock import patch

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
        list(chunks)
        self._state.current_request_id = None
        self._state.state = "idle"


class ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


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


def test_speak_creates_completed_job_record():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()

    with patch("melotts_server.service.threading.Thread", ImmediateThread):
        request_id, _ = service.speak("hello", 1.0, 1.0)

    job = service.get_job(request_id)
    assert job is not None
    assert job["state"] == "completed"


def test_synthesize_rejects_text_over_limit():
    service = TTSService(Settings(max_text_chars=3))
    service.engine = FakeEngine()
    service.player = FakePlayer()

    try:
        service.synthesize("hello", 1.0, 1.0)
    except RuntimeError as exc:
        assert "max length" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_status_advertises_session_streaming():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()

    status = service.status()

    assert status["features"]["session_streaming"] is True


def test_session_commit_with_chunks_completes():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()

    session_id, _ = service.create_session("en", 1.0, 1.0)
    service.enqueue_session_chunk(session_id, "hello", 1.0, 1.0)
    payload = service.commit_session(session_id)

    for _ in range(50):
        session = service.get_session(session_id)
        if session is not None and session["state"] == "completed":
            break
        time.sleep(0.01)

    assert payload["session_id"] == session_id
    session = service.get_session(session_id)
    assert session is not None
    assert session["chunk_count"] == 1
    assert session["state"] == "completed"


def test_session_cancel_marks_cancelled():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()

    session_id, _ = service.create_session("en", 1.0, 1.0)
    payload = service.cancel_session(session_id)

    assert payload["state"] == "cancelled"
    session = service.get_session(session_id)
    assert session is not None
    assert session["state"] == "cancelled"


def test_session_rejects_chunks_after_commit():
    service = TTSService(Settings())
    service.engine = FakeEngine()
    service.player = FakePlayer()

    session_id, _ = service.create_session("en", 1.0, 1.0)
    service.commit_session(session_id)

    try:
        service.enqueue_session_chunk(session_id, "late chunk", 1.0, 1.0)
    except RuntimeError as exc:
        assert "committed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
