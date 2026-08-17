"""Topic Engine thresholds — curriculum decisions, not conversation text."""

from __future__ import annotations

from dataclasses import dataclass


TOPIC_ENGINE_CONFIG = {
    "oneActiveTopic": True,
    "resumePartialTopics": True,
    "useCompletionCriteria": True,
    "markRevisitOnActiveWeakness": True,
}


@dataclass(frozen=True)
class TopicEngineConfig:
    one_active_topic: bool = True
    resume_partial_topics: bool = True
    use_completion_criteria: bool = True
    mark_revisit_on_active_weakness: bool = True

    @classmethod
    def from_mapping(cls, raw: dict | None = None) -> "TopicEngineConfig":
        data = dict(TOPIC_ENGINE_CONFIG)
        if raw:
            data.update(raw)
        defaults = cls()
        return cls(
            one_active_topic=bool(data.get("oneActiveTopic", defaults.one_active_topic)),
            resume_partial_topics=bool(
                data.get("resumePartialTopics", defaults.resume_partial_topics)
            ),
            use_completion_criteria=bool(
                data.get("useCompletionCriteria", defaults.use_completion_criteria)
            ),
            mark_revisit_on_active_weakness=bool(
                data.get("markRevisitOnActiveWeakness", defaults.mark_revisit_on_active_weakness)
            ),
        )


DEFAULT_TOPIC_ENGINE_CONFIG = TopicEngineConfig.from_mapping()
