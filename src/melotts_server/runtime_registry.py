from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import Settings
from .english_runtime import MeloTTSEnglishV3Engine
from .inference import MeloTTSRKNNEngine


@dataclass(slots=True)
class RuntimeProfileStatus:
    enabled: bool
    ready: bool
    model_loaded: bool
    model_dir: str
    runtime_name: str
    detail: str | None = None


@dataclass(slots=True)
class RuntimeProfile:
    enabled: bool
    runtime_name: str
    model_dir: str
    engine: MeloTTSRKNNEngine | MeloTTSEnglishV3Engine
    detail: str | None = None


class RuntimeRegistry:
    """Language-routed runtime registry for Chinese and English profiles."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._profiles = {
            "zh": RuntimeProfile(
                enabled=True,
                runtime_name="melo-rknn-zh",
                model_dir=str(settings.model_dir),
                engine=MeloTTSRKNNEngine(settings),
            ),
            "en": RuntimeProfile(
                enabled=settings.english_enabled,
                runtime_name="melo-english-v3",
                model_dir=str(settings.english_bundle_dir),
                engine=MeloTTSEnglishV3Engine(settings),
            ),
        }

    def warmup_enabled(self) -> None:
        for key, profile in self._profiles.items():
            if not profile.enabled:
                continue
            try:
                profile.engine.warmup()
                profile.detail = None
            except Exception as exc:
                profile.detail = str(exc)
                raise RuntimeError(f"{key} runtime warmup failed: {exc}") from exc

    def unload_all(self) -> None:
        for profile in self._profiles.values():
            profile.engine.unload()

    def resolve_language(self, language: str) -> Literal["zh", "en"]:
        lang = (language or "en").lower()
        if lang in {"zh", "yue"}:
            return "zh"
        return "en"

    def get_engine(self, language: str):
        profile_key = self.resolve_language(language)
        profile = self._profiles[profile_key]
        if not profile.enabled:
            raise RuntimeError(f"{profile_key} runtime is not enabled")
        return profile.engine

    def status(self) -> dict[str, RuntimeProfileStatus]:
        statuses: dict[str, RuntimeProfileStatus] = {}
        for key, profile in self._profiles.items():
            engine = profile.engine
            statuses[key] = RuntimeProfileStatus(
                enabled=bool(profile.enabled),
                ready=bool(profile.enabled and engine.loaded),
                model_loaded=bool(engine.loaded),
                model_dir=str(profile.model_dir),
                runtime_name=str(profile.runtime_name),
                detail=profile.detail,
            )
        return statuses
