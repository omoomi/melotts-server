from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .audio import AplayPlayer
from .config import Settings
from .inference import MeloTTSRKNNEngine, wav_bytes_from_float


@dataclass(slots=True)
class ServiceStatus:
    current_request_id: str | None = None
    state: str = "idle"
    detail: str | None = None


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.engine = MeloTTSRKNNEngine(settings)
        self.player = AplayPlayer(settings.sample_rate, settings.alsa_device, settings.prebuffer_ms)
        self._status = ServiceStatus()
        self._busy_lock = threading.Lock()

    def startup(self) -> None:
        self.engine.warmup()

    def shutdown(self) -> None:
        self.player.stop()
        self.engine.unload()

    def speak(self, text: str, speed: float, volume: float, playback_device: str | None = None) -> tuple[str, datetime]:
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError("service is busy")
        request_id = uuid.uuid4().hex
        queued_at = datetime.now(timezone.utc)
        self._status = ServiceStatus(current_request_id=request_id, state="synthesizing")

        thread = threading.Thread(
            target=self._run_speak_job,
            args=(request_id, text, speed, volume, playback_device),
            daemon=True,
        )
        thread.start()
        return request_id, queued_at

    def synthesize(self, text: str, speed: float, volume: float) -> bytes:
        audio = self.engine.synthesize(text=text, speed=speed, volume=volume)
        return wav_bytes_from_float(audio, self._settings.sample_rate)

    def stop(self) -> None:
        self.player.stop()
        self._status = ServiceStatus(state="stopped", detail="stopped by request")

    def status(self) -> dict[str, object]:
        player_state = self.player.state
        state = player_state.state if player_state.state != "idle" else self._status.state
        detail = player_state.detail or self._status.detail
        return {
            "ready": self.engine.loaded and self.player.available(),
            "model_loaded": self.engine.loaded,
            "audio_ready": self.player.available(),
            "current_request_id": player_state.current_request_id or self._status.current_request_id,
            "state": state,
            "queue_depth": player_state.queue_depth,
            "sample_rate": self._settings.sample_rate,
            "model_dir": str(self._settings.model_dir),
            "audio_device": self.player.device,
            "detail": detail,
        }

    def _run_speak_job(self, request_id: str, text: str, speed: float, volume: float, playback_device: str | None) -> None:
        try:
            chunks = (chunk.audio for chunk in self.engine.synthesize_stream(text=text, speed=speed, volume=volume))
            self.player.play_stream(request_id, chunks, playback_device)
            self._status = ServiceStatus(state="idle")
        except Exception as exc:
            self._status = ServiceStatus(current_request_id=request_id, state="error", detail=str(exc))
            raise
        finally:
            if self._busy_lock.locked():
                try:
                    self._busy_lock.release()
                except RuntimeError:
                    pass
