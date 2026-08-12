"""Mode registry — pick strategy by name without if/else chains."""

from __future__ import annotations

from collections.abc import Callable

from core.config import Settings
from modes.base import TalkMode
from modes.realtime import RealtimeMode
from modes.recorded import RecordedMode
from wrappers.deepgram_stt import DeepgramSTT
from wrappers.media_pipeline import MediaPipeline

ModeFactory = Callable[[Settings, DeepgramSTT, MediaPipeline], TalkMode]

MODE_REGISTRY: dict[str, ModeFactory] = {
    "recorded": RecordedMode,
    "realtime": RealtimeMode,
}


def build_mode(
    name: str,
    settings: Settings,
    stt: DeepgramSTT,
    media: MediaPipeline,
) -> TalkMode:
    factory = MODE_REGISTRY.get(name)
    if factory is None:
        known = ", ".join(sorted(MODE_REGISTRY))
        raise KeyError(f"Unknown TALK_MODE={name!r}. Use one of: {known}")
    return factory(settings, stt, media)
