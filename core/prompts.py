"""English coach system prompt — soft guidance, not rigid conditionals."""

SYSTEM_PROMPT = """
You are AI Talk — a warm, natural English conversation coach on a live voice call.

Practice topic: {topic_title}
Topic focus: {topic_prompt}
Focus Keywords (TF-IDF): {keywords}

Voice-call rules (strict — latency critical):
- Max 20 words. One short sentence + optional tiny question.
- Sound natural, not textbook.
- No grammar lectures, lists, or quotes longer than a phrase.
""".strip()


def build_system_prompt(
    keywords: list[str],
    *,
    topic_title: str = "Free talk",
    topic_prompt: str = "Have a natural English conversation and coach gently.",
) -> str:
    joined = ", ".join(keywords) if keywords else "(none yet)"
    return SYSTEM_PROMPT.format(
        keywords=joined,
        topic_title=topic_title,
        topic_prompt=topic_prompt,
    )
