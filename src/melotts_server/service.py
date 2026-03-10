from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from .audio import AplayPlayer
from .config import Settings
from .inference import MeloTTSRKNNEngine, wav_bytes_from_float


@dataclass(slots=True)
class ServiceStatus:
    current_request_id: str | None = None
    state: str = "idle"
    detail: str | None = None


@dataclass(slots=True)
class JobRecord:
    request_id: str
    state: Literal["accepted", "synthesizing", "playing", "completed", "stopped", "error"]
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.engine = MeloTTSRKNNEngine(settings)
        self.player = AplayPlayer(settings.sample_rate, settings.alsa_device, settings.prebuffer_ms)
        self._status = ServiceStatus()
        self._busy_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._jobs_lock = threading.Lock()

    def startup(self) -> None:
        self.engine.warmup()

    def shutdown(self) -> None:
        self.player.stop()
        self.engine.unload()

    def speak(self, text: str, speed: float, volume: float, playback_device: str | None = None) -> tuple[str, datetime]:
        self._validate_text(text)
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError("service is busy")
        request_id = uuid.uuid4().hex
        queued_at = datetime.now(timezone.utc)
        self._status = ServiceStatus(current_request_id=request_id, state="accepted")
        self._set_job(
            JobRecord(
                request_id=request_id,
                state="accepted",
                queued_at=queued_at,
            )
        )

        thread = threading.Thread(
            target=self._run_speak_job,
            args=(request_id, text, speed, volume, playback_device),
            daemon=True,
        )
        thread.start()
        return request_id, queued_at

    def synthesize(self, text: str, speed: float, volume: float) -> bytes:
        self._validate_text(text)
        audio = self.engine.synthesize(text=text, speed=speed, volume=volume)
        return wav_bytes_from_float(audio, self._settings.sample_rate)

    def stop(self) -> None:
        self.player.stop()
        request_id = self._status.current_request_id
        self._status = ServiceStatus(current_request_id=request_id, state="stopped", detail="stopped by request")
        if request_id is not None:
            self._update_job(
                request_id,
                state="stopped",
                finished_at=datetime.now(timezone.utc),
                detail="stopped by request",
            )

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

    def get_job(self, request_id: str) -> dict[str, object] | None:
        with self._jobs_lock:
            job = self._jobs.get(request_id)
        if job is None:
            return None
        payload = self._serialize_job(job)
        player_state = self.player.state
        if player_state.current_request_id == request_id and player_state.state in {"synthesizing", "playing", "stopped"}:
            payload["state"] = player_state.state
            payload["detail"] = player_state.detail or payload["detail"]
        return payload

    def _run_speak_job(self, request_id: str, text: str, speed: float, volume: float, playback_device: str | None) -> None:
        try:
            started_at = datetime.now(timezone.utc)
            self._status = ServiceStatus(current_request_id=request_id, state="synthesizing")
            self._update_job(request_id, state="synthesizing", started_at=started_at)
            chunks = (chunk.audio for chunk in self.engine.synthesize_stream(text=text, speed=speed, volume=volume))
            self.player.play_stream(request_id, chunks, playback_device)
            finished_at = datetime.now(timezone.utc)
            final_state = "stopped" if self.player.state.state == "stopped" else "completed"
            self._status = ServiceStatus(state="idle")
            self._update_job(request_id, state=final_state, finished_at=finished_at, detail=None)
        except Exception as exc:
            self._status = ServiceStatus(current_request_id=request_id, state="error", detail=str(exc))
            self._update_job(
                request_id,
                state="error",
                finished_at=datetime.now(timezone.utc),
                detail=str(exc),
            )
        finally:
            if self._busy_lock.locked():
                try:
                    self._busy_lock.release()
                except RuntimeError:
                    pass

    def _validate_text(self, text: str) -> None:
        if len(text) > self._settings.max_text_chars:
            raise RuntimeError(f"text exceeds max length of {self._settings.max_text_chars} characters")

    def _set_job(self, job: JobRecord) -> None:
        with self._jobs_lock:
            self._jobs[job.request_id] = job

    def _update_job(self, request_id: str, **updates: object) -> None:
        with self._jobs_lock:
            job = self._jobs.get(request_id)
            if job is None:
                return
            for key, value in updates.items():
                if value is not None:
                    setattr(job, key, value)

    @staticmethod
    def _serialize_job(job: JobRecord) -> dict[str, object]:
        return {
            "request_id": job.request_id,
            "state": job.state,
            "queued_at": job.queued_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "detail": job.detail,
        }
