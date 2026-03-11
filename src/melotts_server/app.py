from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from .config import Settings
from .schemas import (
    AudioDeviceResponse,
    JobResponse,
    JobStatusResponse,
    SessionChunkRequest,
    SessionCreateRequest,
    SessionResponse,
    SessionStatusResponse,
    SpeakRequest,
    StatusResponse,
    SynthesizeRequest,
)
from .service import TTSService


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    service = TTSService(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.startup()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="MeloTTS RK3588 Server", lifespan=lifespan)

    @app.get("/v1/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        return StatusResponse(**service.status())

    @app.get("/v1/audio/devices", response_model=AudioDeviceResponse)
    async def devices() -> AudioDeviceResponse:
        return AudioDeviceResponse(devices=service.player.list_devices())

    @app.post("/v1/speak", response_model=JobResponse)
    async def speak(request: SpeakRequest) -> JobResponse:
        try:
            request_id, queued_at = service.speak(
                text=request.text,
                speed=request.speed,
                volume=request.volume,
                language=request.language,
                playback_device=request.playback_device,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JobResponse(
            request_id=request_id,
            state="accepted",
            queued_at=queued_at.isoformat(),
            estimated_latency_ms=config.prebuffer_ms,
        )

    @app.post("/v1/sessions", response_model=SessionResponse)
    async def create_session(request: SessionCreateRequest) -> SessionResponse:
        try:
            session_id, queued_at = service.create_session(
                language=request.language,
                speed=request.speed,
                volume=request.volume,
                playback_device=request.playback_device,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionResponse(session_id=session_id, state="accepted", queued_at=queued_at.isoformat())

    @app.post("/v1/sessions/{session_id}/chunks", response_model=SessionStatusResponse)
    async def enqueue_session_chunk(session_id: str, request: SessionChunkRequest) -> SessionStatusResponse:
        try:
            payload = service.enqueue_session_chunk(
                session_id=session_id,
                text=request.text,
                speed=request.speed,
                volume=request.volume,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionStatusResponse(**payload)

    @app.post("/v1/sessions/{session_id}/commit", response_model=SessionStatusResponse)
    async def commit_session(session_id: str) -> SessionStatusResponse:
        try:
            payload = service.commit_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionStatusResponse(**payload)

    @app.post("/v1/sessions/{session_id}/cancel", response_model=SessionStatusResponse)
    async def cancel_session(session_id: str) -> SessionStatusResponse:
        try:
            payload = service.cancel_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SessionStatusResponse(**payload)

    @app.get("/v1/sessions/{session_id}", response_model=SessionStatusResponse)
    async def session_status(session_id: str) -> SessionStatusResponse:
        payload = service.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionStatusResponse(**payload)

    @app.post("/v1/synthesize")
    async def synthesize(request: SynthesizeRequest) -> Response:
        try:
            wav_data = service.synthesize(
                text=request.text,
                speed=request.speed,
                volume=request.volume,
                language=request.language,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=wav_data, media_type="audio/wav")

    @app.get("/v1/jobs/{request_id}", response_model=JobStatusResponse)
    async def job_status(request_id: str) -> JobStatusResponse:
        job = service.get_job(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatusResponse(**job)

    @app.post("/v1/stop", response_model=JobResponse)
    async def stop() -> JobResponse:
        service.stop()
        return JobResponse(
            request_id="stop",
            state="accepted",
            queued_at="",
            estimated_latency_ms=0,
            detail="playback stopped",
        )

    return app
