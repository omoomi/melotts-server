from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from .config import Settings
from .schemas import AudioDeviceResponse, JobResponse, SpeakRequest, StatusResponse, SynthesizeRequest
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

    @app.post("/v1/synthesize")
    async def synthesize(request: SynthesizeRequest) -> Response:
        wav_data = service.synthesize(text=request.text, speed=request.speed, volume=request.volume)
        return Response(content=wav_data, media_type="audio/wav")

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

