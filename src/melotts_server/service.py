from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
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
        self.engine = MeloTTSRKNNEngine(settings)
        self.player = AplayPlayer(settings.sample_rate, settings.alsa_device, settings.prebuffer_ms)
        self._status = ServiceStatus()
        self._busy_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._jobs_lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._session_queues: dict[str, Queue[SessionChunk | None]] = {}
        self._sessions_lock = threading.Lock()

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
            self._mark_session_stopped(request_id, "stopped by request")

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
            "features": {"session_streaming": True},
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
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError("service is busy")

        session_id = uuid.uuid4().hex
        queued_at = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id=session_id,
            state="accepted",
            queued_at=queued_at,
            language=language,
            speed=speed,
            volume=volume,
            playback_device=playback_device,
        )
        with self._sessions_lock:
            self._sessions[session_id] = session
            self._session_queues[session_id] = Queue()

        self._status = ServiceStatus(current_request_id=session_id, state="accepted")
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
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, object] | None:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            payload = self._serialize_session(session)
        player_state = self.player.state
        if player_state.current_request_id == session_id and player_state.state in {"synthesizing", "playing", "stopped"}:
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

    def _run_session(self, session_id: str) -> None:
        session = self._get_session_record(session_id)
        queue = self._get_session_queue(session_id)
        if session is None or queue is None:
            self._release_busy_lock()
            return
        try:
            self._status = ServiceStatus(current_request_id=session_id, state="accepted")

            def audio_stream():
                while True:
                    item = queue.get()
                    if item is None:
                        break
                    if session.cancelled:
                        break
                    started_at = datetime.now(timezone.utc)
                    with self._sessions_lock:
                        if session.started_at is None:
                            session.started_at = started_at
                        session.state = "synthesizing"
                        session.pending_chunks = max(0, session.pending_chunks - 1)
                    self._status = ServiceStatus(current_request_id=session_id, state="synthesizing")
                    for chunk in self.engine.synthesize_stream(text=item.text, speed=item.speed, volume=item.volume):
                        if session.cancelled:
                            break
                        yield chunk.audio
                    if session.cancelled:
                        break

            first_item = queue.get()
            if first_item is None:
                with self._sessions_lock:
                    session.state = "completed" if not session.cancelled else "cancelled"
                    session.finished_at = datetime.now(timezone.utc)
                self._status = ServiceStatus(state="idle")
                return

            def combined_stream():
                item = first_item
                if item is not None and not session.cancelled:
                    started_at = datetime.now(timezone.utc)
                    with self._sessions_lock:
                        if session.started_at is None:
                            session.started_at = started_at
                        session.state = "synthesizing"
                        session.pending_chunks = max(0, session.pending_chunks - 1)
                    self._status = ServiceStatus(current_request_id=session_id, state="synthesizing")
                    for chunk in self.engine.synthesize_stream(text=item.text, speed=item.speed, volume=item.volume):
                        if session.cancelled:
                            return
                        yield chunk.audio
                yield from audio_stream()

            self.player.play_stream(session_id, combined_stream(), session.playback_device)
            finished_at = datetime.now(timezone.utc)
            player_state = self.player.state.state
            if session.cancelled:
                final_state = "cancelled"
            elif player_state == "stopped":
                final_state = "stopped"
            else:
                final_state = "completed"
            with self._sessions_lock:
                session.state = final_state
                session.finished_at = finished_at
                session.detail = None if final_state == "completed" else session.detail
            self._status = ServiceStatus(state="idle")
        except Exception as exc:
            with self._sessions_lock:
                session.state = "error"
                session.finished_at = datetime.now(timezone.utc)
                session.detail = str(exc)
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
                if value is not None:
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
