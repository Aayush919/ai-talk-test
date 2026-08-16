"""RAM memory — STM + working + episodic + semantic. Hot path = no network."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.curriculum import get_lesson, next_lesson_id
from core.learn.intel import LearnerIntel
from core.learn.teach import model_for, teach_line
from core.memory.extract import (
    Candidate,
    detect_interests,
    extract_candidates,
    is_fragment,
    is_plausible_hobby,
    is_plausible_job,
    is_plausible_name,
    is_plausible_place,
    is_pushback,
    is_question_sentence,
    name_only_ask,
    phrase_match,
    question_fingerprint,
    question_lines,
    wants_recall,
)

# "place" is one requirement filled by either native or current_city
PLACE_SLOTS = ("current_city", "native")

TOPIC_SLOTS: dict[str, tuple[str, ...]] = {
    "introduction": ("name", "place", "job", "hobby"),
    "family": ("name", "place", "hobby"),
    "daily-routine": ("name", "hobby", "job"),
    "job-interview": ("name", "job"),
    "travel": ("name", "place"),
    "shopping": ("name",),
    "food-order": ("name",),
}

SCENE_TOPICS = frozenset(
    {
        "job-interview",
        "food-order",
        "travel",
        "shopping",
        "phone-call",
        "doctor",
        "hotel-checkin",
        "friends-plans",
    }
)

# Soft discovery order — follow their story, then open the next missing area
DISCOVER_AREAS: tuple[tuple[str, str], ...] = (
    ("name", "what they like to be called"),
    ("english_goal", "why they want to learn English"),
    ("routine", "what a normal day looks like"),
    ("family", "who they spend time with at home"),
    ("job", "their work or studies"),
    ("hobby", "what they enjoy in free time"),
)

DISCOVER_MAX_TURNS = 16
_STRENGTH = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

ACTIVITIES: tuple[str, ...] = (
    "casual conversation",
    "a short personal story",
    "an opinion",
    "a real-life scenario",
    "slightly richer vocabulary",
)

_GENERIC_QS: tuple[str, ...] = (
    "Tell me a bit more about that.",
    "What happened next?",
    "Can you say a little more?",
    "And then what did you do?",
    "How was that for you?",
)


@dataclass
class Fact:
    value: str
    confidence: float = 0.5


@dataclass
class Episode:
    text: str
    turn: int


@dataclass
class MemoryBank:
    learner_id: str
    topic_id: str = "introduction"
    semantic: dict[str, Fact] = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)
    ltm_snippets: list[str] = field(default_factory=list)
    last_coach_question: str = ""
    last_fix: str = ""
    last_fix_turn: int = -1
    turns: int = 0
    # Working state — drives progression so nothing repeats mid-call
    phase: str = "discover"
    asked: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    await_repeat: bool = False
    repeat_target: str = ""
    corrected: list[str] = field(default_factory=list)
    interests: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    last_user: str = ""
    pushed_back: bool = False
    short_streak: int = 0
    intel: LearnerIntel = field(default_factory=LearnerIntel)
    lesson: str = "intro"
    call_warm: bool = False
    filled_now: bool = False
    hold_lesson: bool = False
    _seen: tuple[int, str] = (0, "")

    def __post_init__(self) -> None:
        if self.topic_id in SCENE_TOPICS:
            self.phase = "practice"

    # ---------- facts ----------

    def hydrate_semantic(self, facts: dict[str, str]) -> None:
        for slot, value in facts.items():
            if not value or slot in self.semantic:
                continue
            if slot == "name" and not is_plausible_name(value):
                continue
            if slot in {"native", "current_city"} and not is_plausible_place(value):
                continue
            if slot == "job" and not is_plausible_job(value):
                continue
            if slot == "hobby" and not is_plausible_hobby(value):
                continue
            if slot == "intel":
                self.intel = LearnerIntel.loads(value)
                continue
            if slot == "lesson":
                self.lesson = value.strip() or "intro"
                continue
            if slot == "asked":
                self.asked = [p.strip() for p in value.split("|") if p.strip()][:40]
                continue
            if slot == "interests":
                self._parse_interests(value)
            if slot == "weaknesses":
                for tag in value.split(";"):
                    tag = tag.strip()
                    if tag and tag not in self.errors:
                        self.errors.append(tag)
                self.errors = self.errors[-8:]
            self.semantic[slot] = Fact(value=value, confidence=0.7)
        self._sync_phase(returning=True)

    def observe_user(self, text: str, turn: int) -> None:
        """Consolidate one FINAL user line into semantic + episodic (idempotent)."""
        key = (turn, (text or "").strip().lower())
        if key == self._seen:
            return
        self._seen = key
        self.turns = max(self.turns, turn)
        self.last_fix = ""
        self.last_user = (text or "").strip()
        words = self.last_user.split()
        if len(words) <= 4:
            self.short_streak += 1
        else:
            self.short_streak = 0
        if is_pushback(self.last_user):
            self.pushed_back = True
        elif len(words) >= 8:
            self.pushed_back = False
        self._drop_bad_places()
        applied: list[Candidate] = []
        for cand in extract_candidates(text):
            if self._merge(cand, turn):
                applied.append(cand)
        for label, strength in detect_interests(text):
            self._bump_interest(label, strength)
        self.intel.observe_turn(text, turn, self.interests)
        self.errors = list(self.intel.last_hits) or self.errors
        self.errors = self.errors[-8:]
        if applied:
            note = ", ".join(f"{c.slot}={c.value}" for c in applied[:4])
            self.episodes.append(Episode(f"turn {turn}: {note}", turn))
            self.episodes = self.episodes[-8:]
        self.filled_now = self._applied_current_slots(applied)
        self._sync_phase()
        self._sync_lesson()

    def _sync_phase(self, *, returning: bool = False) -> None:
        if self.phase == "practice":
            return
        if self.topic_id in SCENE_TOPICS:
            self.phase = "practice"
            return
        missing = self.missing_discover()
        if not missing:
            self.phase = "practice"
            return
        if returning and self._val("name") and len(missing) <= 2:
            self.phase = "practice"
            return
        if not returning and self.turns >= DISCOVER_MAX_TURNS:
            self.phase = "practice"

    def fix_for(self, turn: int) -> str:
        """Spoken confirmation is unused — facts stay silent in the profile."""
        return ""

    def _drop_bad_places(self, keep: str = "") -> None:
        for key in ("native", "current_city"):
            fact = self.semantic.get(key)
            if not fact:
                continue
            if not is_plausible_place(fact.value):
                del self.semantic[key]
                continue
            if keep and fact.value.lower() == keep.lower() and key != "current_city":
                # one city in one slot — recap reads current_city first
                if "current_city" in self.semantic:
                    del self.semantic[key]

    def _merge(self, cand: Candidate, turn: int) -> bool:
        old = self.semantic.get(cand.slot)
        if cand.slot == "name" and not is_plausible_name(cand.value):
            return False
        if cand.slot in {"native", "current_city"} and not is_plausible_place(cand.value):
            return False
        if cand.slot == "job" and not is_plausible_job(cand.value):
            return False
        if cand.slot == "hobby" and not is_plausible_hobby(cand.value):
            return False

        if cand.correction:
            self.semantic[cand.slot] = Fact(cand.value, cand.confidence)
            if cand.slot in {"native", "current_city"}:
                self._drop_bad_places(keep=cand.value)
                # Keep recap city consistent
                self.semantic["current_city"] = Fact(cand.value, cand.confidence)
                self.semantic.pop("native", None)
            note = f"turn {turn}: {cand.slot} {cand.rejected or (old.value if old else '?')} -> {cand.value}"
            self.episodes.append(Episode(note, turn))
            self.episodes = self.episodes[-8:]
            return True
        if old is None:
            self.semantic[cand.slot] = Fact(cand.value, cand.confidence)
            if cand.slot in {"native", "current_city"}:
                self._drop_bad_places(keep=cand.value)
            return True
        same = cand.value.lower() == old.value.lower()
        if same and cand.confidence > old.confidence:
            old.confidence = cand.confidence
            return False
        old_bad = (
            (cand.slot == "name" and not is_plausible_name(old.value))
            or (
                cand.slot in {"native", "current_city"}
                and not is_plausible_place(old.value)
            )
            or (cand.slot == "job" and not is_plausible_job(old.value))
        )
        if old_bad:
            self.semantic[cand.slot] = Fact(cand.value, cand.confidence)
            if cand.slot in {"native", "current_city"}:
                self._drop_bad_places(keep=cand.value)
            return True
        # Lock filled slots — STT garbage must not overwrite a solid fact
        if cand.confidence < 0.90:
            return False
        if cand.confidence >= old.confidence:
            self.semantic[cand.slot] = Fact(cand.value, cand.confidence)
            if cand.slot in {"native", "current_city"}:
                self._drop_bad_places(keep=cand.value)
            return True
        return False

    def _parse_interests(self, raw: str) -> None:
        for part in (raw or "").split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                key, strength = part.split(":", 1)
                self._bump_interest(key.strip().lower(), strength.strip().upper())
            else:
                self._bump_interest(part.lower(), "MEDIUM")

    def _bump_interest(self, key: str, strength: str) -> None:
        key = (key or "").strip().lower()
        strength = strength if strength in _STRENGTH else "MEDIUM"
        if not key:
            return
        old = self.interests.get(key, "LOW")
        if old == "MEDIUM" and strength == "MEDIUM":
            self.interests[key] = "HIGH"
            return
        if _STRENGTH[strength] > _STRENGTH.get(old, 0):
            self.interests[key] = strength
        elif key not in self.interests:
            self.interests[key] = strength

    def top_interest(self) -> str:
        if not self.interests:
            hobby = self._val("hobby")
            return hobby.lower() if hobby else ""
        ranked = sorted(
            self.interests.items(),
            key=lambda kv: _STRENGTH.get(kv[1], 0),
            reverse=True,
        )
        return ranked[0][0]

    def interest_line(self) -> str:
        if not self.interests:
            hobby = self._val("hobby")
            return f"{hobby.lower()}:MEDIUM" if hobby else ""
        ranked = sorted(
            self.interests.items(),
            key=lambda kv: _STRENGTH.get(kv[1], 0),
            reverse=True,
        )
        return "; ".join(f"{k}:{v}" for k, v in ranked[:5])

    def display_name(self) -> str:
        name = self._val("name")
        return name.split()[0] if name else ""

    # ---------- progression ----------

    def missing_discover(self) -> list[str]:
        left: list[str] = []
        for slot, _hint in DISCOVER_AREAS:
            if slot == "job" and (self._val("job") or self._val("education")):
                continue
            if slot == "hobby" and (self._val("hobby") or self.interests):
                continue
            if not self._val(slot):
                left.append(slot)
        return left

    def next_area_hint(self) -> str:
        missing = self.missing_discover()
        if not missing:
            return ""
        slot = missing[0]
        for key, hint in DISCOVER_AREAS:
            if key == slot:
                return hint
        return ""

    def next_probe(self) -> str:
        """Fallback angle when the learner stalls — still tied to them."""
        top = self.top_interest()
        if top:
            return f"an opinion or short story about {top}"
        hobby = self._val("hobby")
        if hobby:
            return f"what they enjoy most about {hobby}"
        goal = self._val("english_goal")
        if goal:
            return f"a situation where they need English for {goal}"
        return "what they would like to get better at in English"

    def mark_probe_offered(self) -> None:
        return

    def next_question(self) -> str:
        return ""

    def note_coach_reply(self, coach_text: str) -> None:
        """Remember what was asked so the same question never comes twice."""
        self.last_coach_question = coach_text or ""
        for q in question_lines(coach_text):
            if q not in self.asked:
                self.asked.append(q)
        self.asked = self.asked[-40:]
        if self.call_warm and self.last_user:
            self.call_warm = False
        if self.hold_lesson and not self.filled_now and self.last_user:
            self.hold_lesson = False
            self._advance_if_ready()

    def start_repeat(self, target: str) -> None:
        target = (target or "").strip()
        if not target:
            return
        self.await_repeat = True
        self.repeat_target = target
        if target.lower() not in self.corrected:
            self.corrected.append(target.lower())
        self.corrected = self.corrected[-12:]

    def repeat_peek(self, user_text: str) -> str:
        """Read-only check so speculative drafts never consume the repeat state."""
        if not self.await_repeat or not self.repeat_target:
            return ""
        return "matched" if phrase_match(user_text, self.repeat_target) else "skipped"

    def repeat_clear(self) -> None:
        """One chance only — after a committed turn we always move forward."""
        self.await_repeat = False
        self.repeat_target = ""

    def already_corrected(self, target: str) -> bool:
        return (target or "").strip().lower() in self.corrected

    def _activity_hint(self) -> str:
        if self.phase != "practice" or self.turns < 4 or self.turns % 5:
            return ""
        idx = (self.turns // 5 - 1) % len(ACTIVITIES)
        return ACTIVITIES[idx]

    def _slot_filled(self, slot: str) -> bool:
        if slot == "job":
            return bool(self._val("job") or self._val("education"))
        if slot == "place":
            return bool(self._place())
        return bool(self._val(slot))

    def _needs_intro_model(self) -> bool:
        if self.lesson != "intro":
            return False
        name = self._val("name")
        if not name:
            return False
        t = (self.last_user or "").lower()
        if re.search(r"\bmy name is\b", t):
            return False
        return bool(re.search(r"\bi use\b", t) or re.search(r"\bi am i\b", t))

    def model_sentence(self) -> str:
        return model_for(
            self.last_user,
            name=self._val("name"),
            lesson=self.lesson,
        )

    def _lesson_complete(self) -> bool:
        lesson = get_lesson(self.lesson)
        if not all(self._slot_filled(s) for s in lesson.slots):
            return False
        if self._needs_intro_model():
            return False
        return True

    def _applied_current_slots(self, applied: list[Candidate]) -> bool:
        lesson = get_lesson(self.lesson)
        slots = set(lesson.slots)
        if "job" in slots:
            slots.add("education")
        return any(c.slot in slots for c in applied)

    def _advance_if_ready(self) -> None:
        if self.await_repeat or self._needs_intro_model() or self.filled_now:
            return
        if not self._lesson_complete():
            return
        nxt = next_lesson_id(self.lesson)
        if not nxt:
            return
        self.lesson = nxt
        self.filled_now = False
        self.hold_lesson = False

    def _sync_lesson(self) -> None:
        if self.call_warm or self.await_repeat or self._needs_intro_model():
            return
        if is_fragment(self.last_user):
            return
        if self.filled_now:
            self.hold_lesson = True
            return
        if self.hold_lesson:
            return
        if self._lesson_complete():
            self.hold_lesson = True

    def _q_asked(self, question: str, asked_blob: str = "") -> bool:
        key = question_fingerprint(question)
        if not key:
            return False
        if key in self.asked:
            return True
        blob = asked_blob or " ".join(self.asked).lower()
        if key in blob:
            return True
        words = [w for w in key.split() if len(w) > 3]
        if len(words) >= 3 and all(w in blob for w in words[:3]):
            return True
        return False

    def fresh_question(self) -> str:
        asked_blob = " ".join(self.asked).lower()
        lesson = get_lesson(self.lesson)
        for q in (*lesson.follow_ups, *lesson.questions, *_GENERIC_QS):
            if self._q_asked(q, asked_blob):
                continue
            return q
        return "Say a little more about that."

    def never_ask_line(self) -> str:
        if not self.asked:
            return "-"
        return " | ".join(self.asked[-12:])

    def dedupe_reply(self, coach_text: str) -> str:
        """Drop any question we already asked; add a fresh one if needed."""
        parts = re.split(r"(?<=[.!?])\s+", coach_text or "")
        asked_blob = " ".join(self.asked).lower()
        keep: list[str] = []
        kept_q = False
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if is_question_sentence(part) and self._q_asked(part, asked_blob):
                continue
            keep.append(part)
            if is_question_sentence(part):
                kept_q = True
        body = " ".join(keep).strip()
        teaching = bool(
            re.search(r"\btry it\b", body, re.I)
            or re.search(r"\bsay it like this\b", body, re.I)
        )
        if teaching:
            return body
        if not kept_q:
            nxt = self.fresh_question()
            keep.append(nxt)
        return " ".join(p.strip() for p in keep if p.strip())

    def _seed_question(self) -> str:
        lesson = get_lesson(self.lesson)
        asked_blob = " ".join(self.asked).lower()
        for q in lesson.questions:
            if lesson.id == "intro" and "name" in q.lower() and self._val("name"):
                continue
            if self._q_asked(q, asked_blob):
                continue
            return q
        return ""

    def _follow_up_question(self) -> str:
        lesson = get_lesson(self.lesson)
        asked_blob = " ".join(self.asked).lower()
        for q in lesson.follow_ups:
            if self._q_asked(q, asked_blob):
                continue
            return q
        return self._seed_question()

    def topic_about(self) -> str:
        return get_lesson(self.lesson).about

    def next_about(self) -> str:
        nxt = next_lesson_id(self.lesson)
        return get_lesson(nxt).about if nxt else ""

    def next_goal(self) -> str:
        snippet = (self.last_user or "").strip()[:90]
        lesson = get_lesson(self.lesson)
        seed = self._seed_question()
        model = self.model_sentence()
        name = self._val("name")

        if self.topic_id in SCENE_TOPICS:
            return (
                f"Stay in this scene. React to: {snippet or 'their last line'}. "
                "One in-character follow-up."
            )
        if self.call_warm:
            about = self.topic_about()
            ask = self._follow_up_question() if self._lesson_complete() else self._seed_question()
            if not ask or self._q_asked(ask):
                ask = self.fresh_question()
            if self._val("name"):
                return (
                    f"First turn of this call. They answered how they are: {snippet or 'this line'}. "
                    "Greet back in one short line. Do not correct grammar. "
                    f"Then say we'll continue from last time: {about}. "
                    f"Ask: {ask}. Two sentences total. Never repeat an old question."
                )
            return (
                f"First turn of this call. They answered how they are: {snippet or 'this line'}. "
                "Greet back in one short line. Do not correct grammar. "
                f"Then start {about}. Ask: What is your name? Two sentences total."
            )
        if is_fragment(self.last_user):
            return (
                f"LEVEL: {lesson.title}. Their line is incomplete: {snippet or '(cut off)'}. "
                "Ask them to say the full sentence. Do not copy the broken fragment. "
                "Do not change topic."
            )
        if self.await_repeat:
            fresh = self.fresh_question()
            return (
                f"LEVEL: {lesson.title}. They tried or skipped the model sentence. "
                f"Reply to: {snippet or 'this line'}. Brief praise if they tried. "
                f"Then ask a NEW question: {fresh}. Never repeat an old question. "
                "Do not explain the old sentence again."
            )
        if self.pushed_back:
            return (
                f"They pushed back. React to: {snippet or 'this line'}. "
                f"Then gently return to {lesson.title}."
            )
        if self.filled_now:
            follow = self._follow_up_question() or self.fresh_question()
            saved = self._val(get_lesson(self.lesson).slots[0]) if get_lesson(self.lesson).slots else ""
            return (
                f"LEVEL: {lesson.title}. They JUST answered this level: {snippet}. "
                f"Saved: {saved or 'their answer'}. "
                "React to THAT. Ask one NEW follow-up about THIS same topic. "
                f"Follow-up: {follow}. "
                f"NEVER ask again: {self.never_ask_line()}. "
                "Do NOT jump to the next level."
            )
        if self.hold_lesson:
            follow = self._follow_up_question() or self.fresh_question()
            nxt_about = self.next_about()
            return (
                f"LEVEL: {lesson.title}. They continued: {snippet}. "
                "Reply to THIS line first. Stay on this topic if they are still talking about it. "
                f"{('If that thread is done, you may bridge to ' + nxt_about + '.') if nxt_about else ''} "
                f"Ask a NEW question: {follow}. NEVER ask again: {self.never_ask_line()}."
            )

        bits = [
            f"LEVEL: {lesson.title}.",
            lesson.teach,
            f"Their line: {snippet or '(opening)'}.",
            "Reply to THIS line first. Stay on this level.",
            "If they talked about something else, reply to THAT first, then return to this level.",
            "Do not jump to the next level on this turn.",
            "A name is a PERSON. Never ask what they do WITH their name. "
            "Never treat a name as an object, hobby, or daily activity.",
        ]
        if name:
            bits.append(f"Known name: {name} (person).")
        if model:
            bits.append(
                "CORRECT this turn: yes. Teach. Do not ask a new question. Use this shape: "
                f"{teach_line(snippet, model)}"
            )
        elif self.intel.correct_now:
            bits.append(
                "CORRECT this turn: yes. Teach: You said \"...\". We don't say it like that. "
                'Say it like this: "..." Try it. Then stop. No new question.'
            )
        else:
            bits.append("CORRECT this turn: no, unless their sentence is clearly broken.")
            bits.append(f"NEVER ask again: {self.never_ask_line()}.")
            nxt_q = seed or self.fresh_question()
            bits.append(f"If you need a question, ask this NEW one: {nxt_q}")
        bits.append("One question max. Never a list. Never invent facts. Never repeat a question.")
        return " ".join(bits)

    def background_tick(self) -> None:
        self.intel.background_tick()
        if self.intel.focus:
            self.intel.note_practice(self.intel.focus)

    def summary_line(self) -> str:
        facts = self.persistable_facts()
        has_who = any(
            facts.get(k) for k in ("name", "job", "hobby", "english_goal")
        )
        if not has_who and not self.interests:
            return "(new learner)"
        bits = []
        if facts.get("name"):
            bits.append(facts["name"])
        place = facts.get("current_city") or facts.get("native")
        if place:
            bits.append(f"lives in {place}")
        if facts.get("job"):
            bits.append(f"works as {facts['job']}")
        if facts.get("hobby"):
            bits.append(f"enjoys {facts['hobby']}")
        if facts.get("english_goal"):
            bits.append(f"goal: {facts['english_goal']}")
        interests = self.interest_line()
        if interests:
            bits.append(f"interests {interests}")
        return "; ".join(bits) or "(new learner)"

    # ---------- recall ----------

    def _val(self, slot: str) -> str:
        fact = self.semantic.get(slot)
        if not fact:
            return ""
        if slot == "name" and not is_plausible_name(fact.value):
            return ""
        if slot in {"native", "current_city"} and not is_plausible_place(fact.value):
            return ""
        if slot == "job" and not is_plausible_job(fact.value):
            return ""
        if slot == "hobby" and not is_plausible_hobby(fact.value):
            return ""
        return fact.value

    def _place(self) -> str:
        return self._val("current_city") or self._val("native")

    def spoken_recall(self, user_text: str) -> str | None:
        """Instant recap from RAM — no LLM. None if they did not ask."""
        if not wants_recall(user_text):
            return None
        name = self._val("name")
        if name_only_ask(user_text):
            if name:
                return f"Yes. Your name is {name}."
            return "I didn't catch your name clearly. Please say it slowly."
        native = self._place()
        job = self._val("job")
        hobby = self._val("hobby")
        bits: list[str] = []
        if name:
            bits.append(f"you're {name}")
        if native:
            bits.append(f"from {native}")
        if job:
            bits.append(f"you work as a {job}")
        if hobby:
            bits.append(f"you like {hobby}")
        top = self.top_interest()
        if top and (not hobby or top not in hobby.lower()):
            bits.append(f"you're into {top}")
        if not bits:
            return "I don't have your details yet. Tell me your name again."
        body = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + f", and {bits[-1]}"
        body = body[0].upper() + body[1:]
        return f"Yes. {body}."

    def persistable_facts(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, fact in self.semantic.items():
            if key == "name" and not is_plausible_name(fact.value):
                continue
            if key in {"native", "current_city"} and not is_plausible_place(fact.value):
                continue
            if key == "job" and not is_plausible_job(fact.value):
                continue
            if key == "hobby" and not is_plausible_hobby(fact.value):
                continue
            if key in {"interests", "weaknesses", "intel", "lesson", "asked"}:
                continue
            out[key] = fact.value
        if self.interests:
            out["interests"] = self.interest_line()
        if self.errors:
            out["weaknesses"] = "; ".join(self.errors[-6:])
        if self.topic_id:
            out["recent_topics"] = self.topic_id
        out["intel"] = self.intel.dumps()
        out["lesson"] = self.lesson
        if self.asked:
            out["asked"] = " | ".join(self.asked[-24:])
        return out

    def fact_card(self) -> str:
        facts = self.persistable_facts()
        if not facts:
            return "(none yet)"
        order = (
            "name",
            "native",
            "current_city",
            "job",
            "hobby",
            "english_goal",
            "education",
            "family",
            "routine",
            "interests",
            "weaknesses",
        )
        parts = [f"{k}={facts[k]}" for k in order if k in facts]
        extra = [
            f"{k}={v}"
            for k, v in facts.items()
            if k not in order and k not in {"intel", "lesson", "asked"}
        ]
        extra.append(f"lesson={self.lesson}")
        return "; ".join(parts + extra)

    def episode_line(self) -> str:
        if not self.episodes:
            return "(none)"
        return " | ".join(e.text for e in self.episodes[-4:])

    def prompt_pack(self) -> str:
        """Compact pack — intel card, not the transcript."""
        ltm = "; ".join(self.ltm_snippets[:2]) if self.ltm_snippets else "-"
        asked = " | ".join(self.asked[-5:]) if self.asked else "-"
        lesson = get_lesson(self.lesson)
        return (
            "Silent learner state (never read aloud):\n"
            f"- Who: {self.summary_line()}\n"
            f"- Level: {lesson.title} ({self.lesson})\n"
            f"- Call start: {'greet + how are you, then last topic' if self.call_warm else 'continue the level'}\n"
            f"- Never ask again: {self.never_ask_line()}\n"
            f"- Intel: {self.intel.compact_line()}\n"
            f"- Asked: {asked}\n"
            f"- Turn: {self.next_goal()}\n"
            f"- LTM: {ltm}\n"
        )
