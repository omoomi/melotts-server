from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.1, le=3.0)
    language: str = Field(default="en", min_length=2, max_length=8)
    playback_device: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.1, le=3.0)


class JobResponse(BaseModel):
    request_id: str
    state: Literal["accepted", "busy", "error"]
    queued_at: str
    estimated_latency_ms: int
    detail: Optional[str] = None


class SessionCreateRequest(BaseModel):
    language: str = Field(default="en", min_length=2, max_length=8)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.1, le=3.0)
    playback_device: Optional[str] = None


class SessionChunkRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=8)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.1, le=3.0)


class SessionResponse(BaseModel):
    session_id: str
    state: Literal["accepted", "streaming", "committed", "completed", "stopped", "cancelled", "error"]
    queued_at: str
    detail: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    state: Literal["accepted", "streaming", "synthesizing", "playing", "committed", "completed", "stopped", "cancelled", "error"]
    queued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    chunk_count: int
    pending_chunks: int
    detail: Optional[str] = None


class JobStatusResponse(BaseModel):
    request_id: str
    state: Literal["accepted", "synthesizing", "playing", "completed", "stopped", "error"]
    queued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    detail: Optional[str] = None


class StatusResponse(BaseModel):
    ready: bool
    model_loaded: bool
    audio_ready: bool
    current_request_id: Optional[str]
    state: Literal["idle", "accepted", "synthesizing", "playing", "stopped", "error"]
    queue_depth: int
    sample_rate: int
    model_dir: str
    audio_device: str
    features: dict[str, bool] = Field(default_factory=dict)
    detail: Optional[str] = None


class AudioDeviceResponse(BaseModel):
    devices: list[str]
