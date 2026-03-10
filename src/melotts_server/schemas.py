from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.1, le=3.0)
    speaker_id: Optional[str] = None
    playback_device: Optional[str] = None


class SynthesizeRequest(SpeakRequest):
    format: Literal["wav", "pcm_s16le"] = "wav"


class JobResponse(BaseModel):
    request_id: str
    state: Literal["accepted", "busy", "error"]
    queued_at: str
    estimated_latency_ms: int
    detail: Optional[str] = None


class StatusResponse(BaseModel):
    ready: bool
    model_loaded: bool
    audio_ready: bool
    current_request_id: Optional[str]
    state: Literal["idle", "synthesizing", "playing", "stopped", "error"]
    queue_depth: int
    sample_rate: int
    model_dir: str
    audio_device: str
    detail: Optional[str] = None


class AudioDeviceResponse(BaseModel):
    devices: list[str]

