"""Prompts for runtime conversation decisions — never expose internal state."""

from core.conversation.prompts import GENERATE_SYSTEM_PROMPT, build_generate_user_prompt

__all__ = ["GENERATE_SYSTEM_PROMPT", "build_generate_user_prompt"]
