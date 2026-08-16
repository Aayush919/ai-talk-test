"""Compact learner intelligence — scoring only, no decision trees.

Hot path: tag a turn + EWMA update + one correction flag.
Background: level estimate, mastered skills, compact serialize.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# skill -> (severity 0-1, base confidence 0-1, compiled pattern)
_TAG_BANK: tuple[tuple[str, float, float, re.Pattern[str]], ...] = (
    (
        "past_tense",
        0.72,
        0.78,
        re.compile(
            r"\b(yesterday|last night|last week|last month|\d+\s+days?\s+ago)\b"
            r".{0,40}\b(go|goes|eat|come|buy|see|do|get|take|play)\b",
            re.I,
        ),
    ),
    (
        "am_base",
        0.80,
        0.86,
        re.compile(r"\bi am (go|eat|come|buy|see|do|get|take)\b", re.I),
    ),
    (
        "subj_verb",
        0.68,
        0.70,
        re.compile(r"\b(he|she|it)\s+(go|eat|come|want|like|play|work|live)\b", re.I),
    ),
)

_PAST_OK = re.compile(
    r"\b(went|ate|came|bought|saw|did|got|took|played|was|were)\b", re.I
)

ALPHA_HIT = 0.22  # error evidence
ALPHA_OK = 0.10  # clean evidence
CORRECT_MIN = 0.42  # importance threshold
MASTERED = 0.82
MASTER_MIN_N = 4


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _ewma(prev: float, evidence: float, alpha: float) -> float:
    return _clip((1.0 - alpha) * prev + alpha * evidence)


@dataclass
class SkillStat:
    mastery: float = 0.45
    attempts: int = 0
    errors: int = 0
    last_turn: int = 0
    severity: float = 0.5
    confidence: float = 0.5

    def importance(self) -> float:
        freq = min(1.0, self.errors / 3.0) if self.errors else 0.0
        return _clip(
            self.severity
            * self.confidence
            * (0.35 + 0.65 * freq)
            * (1.0 - self.mastery)
        )


@dataclass
class ErrorHit:
    skill: str
    severity: float
    confidence: float


def tag_turn(text: str) -> list[ErrorHit]:
    """Cheap structured tags. Empty unless a pattern fires."""
    t = text or ""
    if not t.strip():
        return []
    hits: list[ErrorHit] = []
    for skill, sev, conf, pat in _TAG_BANK:
        if not pat.search(t):
            continue
        if skill == "past_tense" and _PAST_OK.search(t):
            continue
        hits.append(ErrorHit(skill, sev, conf))
    return hits


@dataclass
class LearnerIntel:
    level: str = "A2"
    vocab: float = 0.45
    fluency: float = 0.45
    skills: dict[str, SkillStat] = field(default_factory=dict)
    interests: dict[str, float] = field(default_factory=dict)
    practiced: list[str] = field(default_factory=list)
    mastered: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    last_hits: list[str] = field(default_factory=list)
    focus: str = ""
    correct_now: bool = False
    word_ewma: float = 8.0

    def observe_turn(self, text: str, turn: int, interests: dict[str, str] | None = None) -> None:
        """Hot path — microseconds. Updates scores used by the next prompt."""
        words = (text or "").split()
        n = len(words)
        self.word_ewma = _ewma(self.word_ewma, float(n), 0.18)
        self.fluency = _clip(self.word_ewma / 14.0)
        if n >= 6:
            uniq = len({w.lower() for w in words if len(w) > 3})
            self.vocab = _ewma(self.vocab, _clip(uniq / max(4.0, n * 0.45)), 0.12)

        if interests:
            for key, strength in interests.items():
                bump = {"HIGH": 0.85, "MEDIUM": 0.55, "LOW": 0.25}.get(strength, 0.4)
                old = self.interests.get(key, 0.3)
                self.interests[key] = _ewma(old, bump, 0.25)

        hits = tag_turn(text)
        hit_ids = {h.skill for h in hits}
        self.last_hits = list(hit_ids)

        for h in hits:
            st = self.skills.setdefault(h.skill, SkillStat())
            st.severity = max(st.severity, h.severity)
            st.confidence = max(st.confidence, h.confidence)
            st.attempts += 1
            st.errors += 1
            st.last_turn = turn
            st.mastery = _ewma(st.mastery, 0.0, ALPHA_HIT)

        # Weak evidence of control: a longer clean line slightly raises open skills
        if not hits and n >= 5:
            for st in self.skills.values():
                if turn - st.last_turn <= 8:
                    st.attempts += 1
                    st.mastery = _ewma(st.mastery, 1.0, ALPHA_OK)

        self.correct_now = False
        self.focus = ""
        best: SkillStat | None = None
        best_id = ""
        best_imp = 0.0
        for sid, st in self.skills.items():
            imp = st.importance()
            if sid in hit_ids and imp > best_imp:
                best_imp = imp
                best = st
                best_id = sid
        if best and best_imp >= CORRECT_MIN and best_id not in self.corrections[-2:]:
            self.correct_now = True
            self.focus = best_id
        elif best_id:
            self.focus = best_id
        else:
            self.focus = self._priority_focus()

    def _priority_focus(self) -> str:
        """Next practice angle: interest vs weakness vs novelty. No checklist."""
        scored: list[tuple[float, str]] = []
        for sid, st in self.skills.items():
            if st.mastery >= MASTERED and st.attempts >= MASTER_MIN_N:
                continue
            scored.append((st.importance() + 0.15 * (1.0 - st.mastery), f"skill:{sid}"))
        for topic, strength in self.interests.items():
            recency_pen = 0.2 if topic in self.practiced[-3:] else 0.0
            scored.append((0.55 * strength - recency_pen, f"interest:{topic}"))
        if not scored:
            return "follow"
        scored.sort(reverse=True)
        return scored[0][1]

    def note_practice(self, label: str) -> None:
        lab = (label or "").strip()
        if not lab:
            return
        if lab not in self.practiced:
            self.practiced.append(lab)
        self.practiced = self.practiced[-10:]

    def note_correction(self, skill: str) -> None:
        if skill and skill not in self.corrections[-1:]:
            self.corrections.append(skill)
        self.corrections = self.corrections[-12:]

    def background_tick(self) -> None:
        """After the turn — not on the LLM/TTS path."""
        mastered: list[str] = []
        weak: list[str] = []
        for sid, st in self.skills.items():
            if st.mastery >= MASTERED and st.attempts >= MASTER_MIN_N:
                mastered.append(sid)
            elif st.importance() >= 0.35:
                weak.append(sid)
        self.mastered = mastered[-8:]
        mix = 0.5 * self.fluency + 0.5 * (
            sum(s.mastery for s in self.skills.values()) / max(1, len(self.skills))
            if self.skills
            else self.vocab
        )
        if mix < 0.32:
            self.level = "A1"
        elif mix < 0.48:
            self.level = "A2"
        elif mix < 0.68:
            self.level = "B1"
        else:
            self.level = "B2"

    def compact_line(self) -> str:
        """Tiny card for the LLM. Not a transcript."""
        weak = []
        for sid, st in sorted(
            self.skills.items(), key=lambda kv: kv[1].importance(), reverse=True
        )[:3]:
            weak.append(f"{sid}:{st.mastery:.2f}/{st.errors}x")
        ints = ",".join(
            f"{k}:{v:.2f}"
            for k, v in sorted(self.interests.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )
        return (
            f"lvl={self.level} flu={self.fluency:.2f} voc={self.vocab:.2f} "
            f"correct={'yes:' + self.focus if self.correct_now else 'no'} "
            f"focus={self.focus or 'follow'} "
            f"weak={';'.join(weak) or '-'} "
            f"mastered={','.join(self.mastered) or '-'} "
            f"int={ints or '-'}"
        )

    def dumps(self) -> str:
        payload = {
            "level": self.level,
            "vocab": round(self.vocab, 3),
            "fluency": round(self.fluency, 3),
            "word_ewma": round(self.word_ewma, 2),
            "focus": self.focus,
            "interests": {k: round(v, 3) for k, v in self.interests.items()},
            "practiced": self.practiced[-10:],
            "mastered": self.mastered,
            "corrections": self.corrections[-12:],
            "skills": {
                k: {
                    "m": round(s.mastery, 3),
                    "n": s.attempts,
                    "e": s.errors,
                    "t": s.last_turn,
                    "sv": round(s.severity, 3),
                    "cf": round(s.confidence, 3),
                }
                for k, s in self.skills.items()
            },
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def loads(cls, raw: str) -> LearnerIntel:
        out = cls()
        if not (raw or "").strip():
            return out
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return out
        out.level = str(data.get("level") or "A2")
        out.vocab = float(data.get("vocab") or 0.45)
        out.fluency = float(data.get("fluency") or 0.45)
        out.word_ewma = float(data.get("word_ewma") or 8.0)
        out.focus = str(data.get("focus") or "")
        ints = data.get("interests") or {}
        if isinstance(ints, dict):
            out.interests = {str(k): float(v) for k, v in ints.items()}
        out.practiced = [str(x) for x in (data.get("practiced") or [])][-10:]
        out.mastered = [str(x) for x in (data.get("mastered") or [])][-8:]
        out.corrections = [str(x) for x in (data.get("corrections") or [])][-12:]
        skills = data.get("skills") or {}
        if isinstance(skills, dict):
            for key, row in skills.items():
                if not isinstance(row, dict):
                    continue
                out.skills[str(key)] = SkillStat(
                    mastery=float(row.get("m") or 0.45),
                    attempts=int(row.get("n") or 0),
                    errors=int(row.get("e") or 0),
                    last_turn=int(row.get("t") or 0),
                    severity=float(row.get("sv") or 0.5),
                    confidence=float(row.get("cf") or 0.5),
                )
        return out
