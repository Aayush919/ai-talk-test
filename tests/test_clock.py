from datetime import datetime

from core.clock import LOCAL_TZ, local_now_line
from core.conversation.prompts import build_generate_user_prompt
from core.prompts import build_system_prompt


def test_local_now_line_uses_india_clock():
    now = datetime(2026, 8, 19, 23, 2, tzinfo=LOCAL_TZ)
    line = local_now_line(now=now)
    assert "Wednesday" in line
    assert "19 August 2026" in line
    assert "11:02 PM" in line
    assert "(night)" in line
    assert "India" in line


def test_live_and_fallback_prompts_include_clock():
    now = datetime(2026, 8, 19, 9, 15, tzinfo=LOCAL_TZ)
    live = build_generate_user_prompt(
        {"topicTitle": "Daily Routine", "currentGoalId": "wake_up"},
        "Hello",
        now=now,
    )
    assert live.startswith("Local now (India):")
    assert "morning" in live
    fallback = build_system_prompt(["wake"])
    assert "Local now (India):" in fallback
