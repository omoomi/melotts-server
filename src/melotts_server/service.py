from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from typing import Literal

from .audio import AplayPlayer
from .config import Settings
from .inference import wav_bytes_from_float
from .runtime_registry import RuntimeRegistry

logger = logging.getLogger(__name__)


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
    language: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None


@dataclass(slots=True)
class SessionChunk:
    text: str
    speed: float
    volume: float


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    state: Literal["accepted", "streaming", "synthesizing", "playing", "committed", "completed", "stopped", "cancelled", "error"]
    queued_at: datetime
    language: str
    speed: float
    volume: float
    playback_device: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    chunk_count: int = 0
    pending_chunks: int = 0
    committed: bool = False
    cancelled: bool = False


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.runtimes = RuntimeRegistry(settings)
        self.player = AplayPlayer(settings.sample_rate, settings.alsa_device, settings.prebuffer_ms)
        self._status = ServiceStatus()
        self._busy_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._jobs_lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._session_queues: dict[str, Queue[SessionChunk | None]] = {}
        self._sessions_lock = threading.Lock()

    def startup(self) -> None:
        self.runtimes.warmup_enabled()

    def shutdown(self) -> None:
        self.player.stop()
        self.runtimes.unload_all()

    def speak(
        self,
        text: str,
        speed: float,
        volume: float,
        language: str,
        playback_device: str | None = None,
    ) -> tuple[str, datetime]:
        self._validate_text(text)
        profile_language = self.runtimes.resolve_language(language)
        self.runtimes.get_engine(profile_language)
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError("service is busy")
        request_id = uuid.uuid4().hex
        queued_at = datetime.now(timezone.utc)
        self._status = ServiceStatus(current_request_id=request_id, state="accepted")
        logger.info("Accepted speak job %s for language=%s text=%r", request_id, profile_language, text[:80])
        self._set_job(
            JobRecord(
                request_id=request_id,
                state="accepted",
                queued_at=queued_at,
                language=profile_language,
            )
        )

        thread = threading.Thread(
            target=self._run_speak_job,
            args=(request_id, text, speed, volume, profile_language, playback_device),
            daemon=True,
        )
        thread.start()
        return request_id, queued_at

    def synthesize(self, text: str, speed: float, volume: float, language: str = "zh") -> bytes:
        self._validate_text(text)
        engine = self.runtimes.get_engine(self.runtimes.resolve_language(language))
        audio = engine.synthesize(text=text, speed=speed, volume=volume)
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
            self._mark_session_stopped(request_id, "stopped by request")

    def status(self) -> dict[str, object]:
        player_state = self.player.state
        state = player_state.state if player_state.state != "idle" else self._status.state
        detail = player_state.detail or self._status.detail
        runtime_statuses = self.runtimes.status()
        profiles = {
            key: {
                "enabled": status.enabled,
                "ready": status.ready,
                "model_loaded": status.model_loaded,
                "model_dir": status.model_dir,
                "runtime_name": status.runtime_name,
                "detail": status.detail,
            }
            for key, status in runtime_statuses.items()
        }
        enabled_profiles = [status for status in runtime_statuses.values() if status.enabled]
        all_loaded = all(status.model_loaded for status in enabled_profiles)
        all_ready = all(status.ready for status in enabled_profiles)
        return {
            "ready": all_ready and self.player.available(),
            "model_loaded": all_loaded,
            "audio_ready": self.player.available(),
            "current_request_id": player_state.current_request_id or self._status.current_request_id,
            "state": state,
            "queue_depth": player_state.queue_depth,
            "sample_rate": self._settings.sample_rate,
            "model_dir": str(self._settings.model_dir),
            "audio_device": self.player.device,
            "profiles": profiles,
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

    def create_session(
        self,
        language: str,
        speed: float,
        volume: float,
        playback_device: str | None = None,
    ) -> tuple[str, datetime]:
        profile_language = self.runtimes.resolve_language(language)
        self.runtimes.get_engine(profile_language)
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError("service is busy")

        session_id = uuid.uuid4().hex
        queued_at = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id=session_id,
            state="accepted",
            queued_at=queued_at,
            language=profile_language,
            speed=speed,
            volume=volume,
            playback_device=playback_device,
        )
        with self._sessions_lock:
            self._sessions[session_id] = session
            self._session_queues[session_id] = Queue()

        self._status = ServiceStatus(current_request_id=session_id, state="accepted")
        logger.info("Created session %s for language=%s", session_id, profile_language)
        thread = threading.Thread(target=self._run_session, args=(session_id,), daemon=True)
        thread.start()
        return session_id, queued_at

    def enqueue_session_chunk(
        self,
        session_id: str,
        text: str,
        speed: float,
        volume: float,
    ) -> dict[str, object]:
        self._validate_text(text)
        session = self._get_session_record(session_id)
        if session is None:
            raise KeyError("session not found")
        if session.cancelled:
            raise RuntimeError("session cancelled")
        if session.committed:
            raise RuntimeError("session already committed")
        if session.state in {"completed", "stopped", "error"}:
            raise RuntimeError(f"session is {session.state}")

        queue = self._get_session_queue(session_id)
        if queue is None:
            raise KeyError("session queue missing")
        queue.put(SessionChunk(text=text, speed=speed, volume=volume))
        logger.info("Enqueued session chunk %s language=%s text=%r", session_id, session.language, text[:120])
        with self._sessions_lock:
            session.chunk_count += 1
            session.pending_chunks += 1
            if session.state == "accepted":
                session.state = "streaming"
        return self.get_session(session_id) or {}

    def commit_session(self, session_id: str) -> dict[str, object]:
        session = self._get_session_record(session_id)
        if session is None:
            raise KeyError("session not found")
        if session.cancelled:
            raise RuntimeError("session cancelled")
        if not session.committed:
            with self._sessions_lock:
                session.committed = True
                if session.state in {"accepted", "streaming"}:
                    session.state = "committed"
            queue = self._get_session_queue(session_id)
            if queue is not None:
                queue.put(None)
            logger.info("Committed session %s with %d chunks", session_id, session.chunk_count)
        return self.get_session(session_id) or {}

    def cancel_session(self, session_id: str) -> dict[str, object]:
        session = self._get_session_record(session_id)
        if session is None:
            raise KeyError("session not found")
        with self._sessions_lock:
            session.cancelled = True
            session.state = "cancelled"
            session.finished_at = datetime.now(timezone.utc)
            session.detail = "cancelled by request"
        queue = self._get_session_queue(session_id)
        if queue is not None:
            queue.put(None)
        self.stop()
        logger.info("Cancelled session %s", session_id)
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, object] | None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            payload = self._serialize_session(session)
        player_state = self.player.state
        if (
            payload["state"] not in {"completed", "cancelled", "error"}
            and player_state.current_request_id == session_id
            and player_state.state in {"synthesizing", "playing", "stopped"}
        ):
            payload["state"] = player_state.state
            payload["detail"] = player_state.detail or payload["detail"]
        return payload

    def _run_speak_job(
        self,
        request_id: str,
        text: str,
        speed: float,
        volume: float,
        language: str,
        playback_device: str | None,
    ) -> None:
        engine = self.runtimes.get_engine(language)
        try:
            started_at = datetime.now(timezone.utc)
            self._status = ServiceStatus(current_request_id=request_id, state="synthesizing")
            self._update_job(request_id, state="synthesizing", started_at=started_at)
            chunks = (chunk.audio for chunk in engine.synthesize_stream(text=text, speed=speed, volume=volume))
            self.player.play_stream(request_id, chunks, playback_device)
            logger.info("Completed speak job %s", request_id)
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
            self._release_busy_lock()

    def _run_session(self, session_id: str) -> None:
        session = self._get_session_record(session_id)
        queue = self._get_session_queue(session_id)
        if session is None or queue is None:
            self._release_busy_lock()
            return
        engine = self.runtimes.get_engine(session.language)
        try:
            self._status = ServiceStatus(current_request_id=session_id, state="accepted")
            self.player.play_stream(
                session_id,
                self._session_audio_stream(session_id, session, queue, engine),
                session.playback_device,
            )
            logger.info("Completed session %s", session_id)
            player_state = self.player.state.state
            if session.cancelled:
                final_state = "cancelled"
            elif player_state == "stopped":
                final_state = "stopped"
            else:
                final_state = "completed"
            self._finish_session(session_id, final_state)
            self._status = ServiceStatus(state="idle")
        except Exception as exc:
            self._finish_session(session_id, "error", str(exc))
            self._status = ServiceStatus(current_request_id=session_id, state="error", detail=str(exc))
        finally:
            self._release_busy_lock()

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
                setattr(job, key, value)

    def _get_session_record(self, session_id: str) -> SessionRecord | None:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def _get_session_queue(self, session_id: str) -> Queue[SessionChunk | None] | None:
        with self._sessions_lock:
            return self._session_queues.get(session_id)

    def _mark_session_stopped(self, session_id: str, detail: str) -> None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if session.state not in {"completed", "cancelled", "error"}:
                session.state = "stopped"
                session.finished_at = datetime.now(timezone.utc)
                session.detail = detail

    def _session_audio_stream(self, session_id: str, session: SessionRecord, queue: Queue[SessionChunk | None], engine):
        while True:
            item = queue.get()
            if item is None or session.cancelled:
                return
            self._mark_session_started(session_id)
            for chunk in engine.synthesize_stream(text=item.text, speed=item.speed, volume=item.volume):
                if session.cancelled:
                    return
                yield chunk.audio

    def _mark_session_started(self, session_id: str) -> None:
        started_at = datetime.now(timezone.utc)
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if session.started_at is None:
                session.started_at = started_at
            session.state = "synthesizing"
            session.pending_chunks = max(0, session.pending_chunks - 1)
        self._status = ServiceStatus(current_request_id=session_id, state="synthesizing")

    def _finish_session(self, session_id: str, state: str, detail: str | None = None) -> None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.state = state
            session.finished_at = datetime.now(timezone.utc)
            if detail is not None or state == "completed":
                session.detail = detail

    def _release_busy_lock(self) -> None:
        if self._busy_lock.locked():
            try:
                self._busy_lock.release()
            except RuntimeError:
                pass

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

    @staticmethod
    def _serialize_session(session: SessionRecord) -> dict[str, object]:
        return {
            "session_id": session.session_id,
            "state": session.state,
            "queued_at": session.queued_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "finished_at": session.finished_at.isoformat() if session.finished_at else None,
            "chunk_count": session.chunk_count,
            "pending_chunks": session.pending_chunks,
            "detail": session.detail,
        }
