from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from melotts_server.config import Settings
from melotts_server.schemas import SessionChunkRequest
from melotts_server.runtime_registry import RuntimeProfileStatus
from melotts_server.service import TTSService


class FakeEngine:
    def __init__(self, name: str):
        self.name = name
        self.loaded = True
        self.calls: list[tuple[str, float, float]] = []

    def warmup(self):
        return None

    def unload(self):
        return None

    def synthesize(self, text: str, speed: float, volume: float):
        self.calls.append((text, speed, volume))
        return np.array([0.0, 0.25], dtype=np.float32)

    def synthesize_stream(self, text: str, speed: float, volume: float):
        self.calls.append((text, speed, volume))
        yield type("Chunk", (), {"audio": np.array([0.0, 0.1], dtype=np.float32)})()


class FakeRuntimeRegistry:
    def __init__(self):
        self.zh = FakeEngine("zh")
        self.en = FakeEngine("en")
        self._profiles = {
            "zh": RuntimeProfileStatus(
                enabled=True,
                ready=True,
                model_loaded=True,
                model_dir="/models/zh",
                runtime_name="zh-runtime",
            ),
            "en": RuntimeProfileStatus(
                enabled=True,
                ready=True,
                model_loaded=True,
                model_dir="/models/en",
                runtime_name="en-runtime",
            ),
        }

    def warmup_enabled(self):
        return None

    def unload_all(self):
        return None

    def resolve_language(self, language: str) -> str:
        return "zh" if language in {"zh", "yue"} else "en"

    def get_engine(self, language: str):
        resolved = self.resolve_language(language)
        if resolved == "zh":
            return self.zh
        if resolved == "en":
            return self.en
        raise RuntimeError(f"unsupported language: {language}")

    def status(self):
        return self._profiles


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


def build_service() -> TTSService:
    service = TTSService(Settings())
    service.runtimes = FakeRuntimeRegistry()
    service.player = FakePlayer()
    return service


def test_synthesize_returns_wav_bytes():
    service = build_service()
    result = service.synthesize("hello", 1.0, 1.0, language="en")
    assert result[:4] == b"RIFF"
    assert service.runtimes.en.calls[0][0] == "hello"


def test_status_ready_when_profiles_and_player_available():
    service = build_service()
    status = service.status()
    assert status["ready"] is True
    assert status["profiles"]["en"]["ready"] is True
    assert status["profiles"]["zh"]["runtime_name"] == "zh-runtime"


def test_speak_routes_english_to_english_runtime():
    service = build_service()

    with patch("melotts_server.service.threading.Thread", ImmediateThread):
        request_id, _ = service.speak("hello", 1.0, 1.0, language="en")

    job = service.get_job(request_id)
    assert job is not None
    assert job["state"] == "completed"
    assert service.runtimes.en.calls[0][0] == "hello"
    assert service.runtimes.zh.calls == []


def test_speak_routes_cantonese_to_chinese_runtime():
    service = build_service()

    with patch("melotts_server.service.threading.Thread", ImmediateThread):
        request_id, _ = service.speak("nei hou", 1.0, 1.0, language="yue")

    job = service.get_job(request_id)
    assert job is not None
    assert job["state"] == "completed"
    assert service.runtimes.zh.calls[0][0] == "nei hou"
    assert service.runtimes.en.calls == []


def test_synthesize_rejects_text_over_limit():
    service = TTSService(Settings(max_text_chars=3))
    service.runtimes = FakeRuntimeRegistry()
    service.player = FakePlayer()

    try:
        service.synthesize("hello", 1.0, 1.0)
    except RuntimeError as exc:
        assert "max length" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_status_reports_runtime_profiles():
    service = build_service()
    status = service.status()
    assert "features" not in status
    assert status["profiles"]["en"]["enabled"] is True


def test_session_commit_with_chunks_completes():
    service = build_service()
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
    assert service.runtimes.en.calls[0][0] == "hello"


def test_session_cancel_marks_cancelled():
    service = build_service()
    session_id, _ = service.create_session("en", 1.0, 1.0)
    payload = service.cancel_session(session_id)

    assert payload["state"] == "cancelled"
    session = service.get_session(session_id)
    assert session is not None
    assert session["state"] == "cancelled"


def test_session_rejects_chunks_after_commit():
    service = build_service()
    session_id, _ = service.create_session("en", 1.0, 1.0)
    service.commit_session(session_id)

    try:
        service.enqueue_session_chunk(session_id, "late chunk", 1.0, 1.0)
    except RuntimeError as exc:
        assert "committed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_session_chunk_request_rejects_language_override():
    try:
        SessionChunkRequest(text="hello", speed=1.0, volume=1.0, language="zh")
    except ValidationError as exc:
        assert "language" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_runtime_resolution_uses_chinese_profile_for_yue():
    service = build_service()
    session_id, _ = service.create_session("yue", 1.0, 1.0)
    session = service.get_session(session_id)

    assert session is not None
    assert service.runtimes.resolve_language("yue") == "zh"
    assert service._sessions[session_id].language == "zh"
