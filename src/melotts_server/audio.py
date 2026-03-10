from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, Optional

import numpy as np


@dataclass(slots=True)
class PlaybackState:
    current_request_id: str | None = None
    state: str = "idle"
    queue_depth: int = 0
    detail: str | None = None


class AplayPlayer:
    def __init__(self, sample_rate: int, device: str, prebuffer_ms: int = 200) -> None:
        self._sample_rate = sample_rate
        self._device = device
        self._prebuffer_ms = prebuffer_ms
        self._stop_event = threading.Event()
        self._state = PlaybackState()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return PlaybackState(
                current_request_id=self._state.current_request_id,
                state=self._state.state,
                queue_depth=self._state.queue_depth,
                detail=self._state.detail,
            )

    def available(self) -> bool:
        return shutil.which("aplay") is not None

    @property
    def device(self) -> str:
        return self._resolve_device(self._device)

    def list_devices(self) -> list[str]:
        if not self.available():
            return []
        result = subprocess.run(["aplay", "-L"], capture_output=True, text=True, check=False)
        return [line.strip() for line in result.stdout.splitlines() if line and not line.startswith(" ")]

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._state.state = "stopped"
            self._state.detail = "playback stopped"
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def play_stream(self, request_id: str, chunks: Iterable[np.ndarray], device: Optional[str] = None) -> None:
        if not self.available():
            raise RuntimeError("aplay is not available on PATH")

        self._stop_event.clear()
        with self._lock:
            self._state.current_request_id = request_id
            self._state.state = "synthesizing"
            self._state.detail = None

        self._play_worker(request_id, chunks, self._resolve_device(device or self._device))

    def _play_worker(self, request_id: str, chunks: Iterable[np.ndarray], device: str) -> None:
        queue: Queue[np.ndarray | None] = Queue(maxsize=8)
        producer_error: list[Exception] = []

        def produce() -> None:
            try:
                for chunk in chunks:
                    if self._stop_event.is_set():
                        break
                    queue.put(chunk)
                    with self._lock:
                        self._state.queue_depth = queue.qsize()
            except Exception as exc:
                producer_error.append(exc)
            finally:
                queue.put(None)
                with self._lock:
                    self._state.queue_depth = queue.qsize()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        command = [
            "aplay",
            "-D",
            device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(self._sample_rate),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        with self._lock:
            self._process = process
        try:
            buffered_samples = 0
            prebuffer_samples = int(self._sample_rate * (self._prebuffer_ms / 1000.0))
            pending: list[np.ndarray] = []
            started = False
            while True:
                chunk = queue.get()
                with self._lock:
                    self._state.queue_depth = queue.qsize()
                if chunk is None:
                    break
                if self._stop_event.is_set():
                    break
                buffered_samples += len(chunk)
                pending.append(chunk)
                if not started and buffered_samples >= prebuffer_samples:
                    with self._lock:
                        self._state.state = "playing"
                    started = True
                    for buffered in pending:
                        self._write_chunk(process, buffered)
                    pending.clear()
                elif started:
                    self._write_chunk(process, chunk)
            if not started:
                with self._lock:
                    self._state.state = "playing"
                for buffered in pending:
                    self._write_chunk(process, buffered)
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait(timeout=10)
            if producer_error:
                raise producer_error[0]
            if return_code != 0 and not self._stop_event.is_set():
                raise RuntimeError(f"aplay exited with code {return_code}")
            with self._lock:
                self._state.state = "idle" if not self._stop_event.is_set() else "stopped"
                self._state.current_request_id = None
                self._state.queue_depth = 0
                self._state.detail = None if not self._stop_event.is_set() else "playback stopped"
        finally:
            if process.poll() is None:
                process.terminate()
            with self._lock:
                self._process = None

    @staticmethod
    def _write_chunk(process: subprocess.Popen[bytes], chunk: np.ndarray) -> None:
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16).tobytes()
        if process.stdin is None:
            raise RuntimeError("aplay stdin unavailable")
        process.stdin.write(pcm)
        process.stdin.flush()

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        devices = self.list_devices()
        for candidate in (
            "sysdefault:CARD=rockchipes8388",
            "plughw:CARD=rockchipes8388,DEV=0",
            "hw:CARD=rockchipes8388,DEV=0",
            "default",
        ):
            if candidate in devices or candidate == "default":
                return candidate
        return "default"
