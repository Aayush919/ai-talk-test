"""Slot consolidation — pattern bank + confidence ranking (not ad-hoc if/else)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.text_clean import clean_speech_text

# not OLD, I am NEW
_CORR_NOT_THEN_NEW = re.compile(
    r"(?:i am not|i'm not|not)\s+([a-z][a-z]+)[,.]?\s*"
    r"(?:i am|i'm|it's|its|my name is)\s+([a-z][a-z]+(?:\s+[a-z]+)?)",
    re.I,
)
# it's NEW not OLD
_CORR_NEW_NOT_OLD = re.compile(
    r"(?:it's|its|it is)\s+([a-z][a-z]+(?:\s+[a-z]+)?)\s+not\s+([a-z][a-z]+)",
    re.I,
)
# No. I am Ayush Malvia
_CORR_NO_I_AM = re.compile(
    r"\bno[,.]?\s+i(?: am|'m)\s+([a-z]{3,}(?:\s+[a-z]{2,})?)",
    re.I,
)
_RECALL_ASK = re.compile(
    r"\b(do you know|know about me|tell me about me|tell me something about me|"
    r"how many thing|what you know|know my name|my name)\b",
    re.I,
)
_NAME_ONLY_ASK = re.compile(r"\b(know )?my name\b", re.I)
_TRY_CUE = re.compile(
    r"\b(try it|try that|try saying|repeat|say it|say that|once more|after me)\b",
    re.I,
)
_ASK_CUE = re.compile(
    r"\b(do you|did you|can you|could you|what|which|how|remember|tell me|"
    r"know about|you know)\b|\?",
    re.I,
)

# (slot, pattern, group, base_confidence)
_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str], int, float], ...] = (
    (
        "name",
        re.compile(
            r"\bmy name(?:\s+is)?\s+(?:i am\s+|i'm\s+|i use\s+)*"
            r"([a-z]{3,}(?:\s+[a-z]{2,})?)",
            re.I,
        ),
        1,
        0.93,
    ),
    (
        "name",
        re.compile(
            r"\bi use\s+([a-z]{3,}(?:\s+[a-z]{2,})?)",
            re.I,
        ),
        1,
        0.88,
    ),
    (
        "name",
        re.compile(
            r"\bi(?:'m| am)\s+(?!a\b|an\b|the\b|from\b|doing\b|not\b|working\b|"
            r"playing\b|living\b|like\b|live\b|good\b|better\b)"
            r"([a-z]{4,})(?:\s+([a-z]{3,}))?",
            re.I,
        ),
        0,
        0.72,
    ),
    ("native", re.compile(r"\b(?:i am |i'm )?from\s+([a-z][a-z]+)", re.I), 1, 0.88),
    ("native", re.compile(r"\bnative(?: place)?(?: is)?\s+([a-z][a-z]+)", re.I), 1, 0.9),
    ("current_city", re.compile(r"\b(?:live|living|stay|staying)\s+in\s+([a-z][a-z]+)", re.I), 1, 0.88),
    ("current_city", re.compile(r"\bin\s+([a-z]{4,})\s+right now", re.I), 1, 0.8),
    (
        "job",
        re.compile(
            r"\b(?:work as(?: a)?|working as(?: a)?|my job is(?: a)?)\s+"
            r"([a-z][a-z ]{2,40}?)(?:\s+job|\s+in\b|[.,!?]|$)",
            re.I,
        ),
        1,
        0.9,
    ),
    (
        "job",
        re.compile(
            r"\bi(?:'m| am) a(?:n)?\s+"
            r"(developer|engineer|student|teacher|doctor|nurse|designer|manager|"
            r"founder|intern|analyst|writer|lawyer|accountant|consultant|"
            r"programmer|freelancer|entrepreneur|homemaker|farmer|driver|chef|"
            r"artist|salesperson|software engineer|data scientist)\b",
            re.I,
        ),
        1,
        0.88,
    ),
    (
        "english_goal",
        re.compile(
            r"\b(?:i want to|i need to|i'm learning english to|learn english (?:to|for)|"
            r"improve my english (?:to|for)|english for)\s+([a-z][a-z ]{2,40})",
            re.I,
        ),
        1,
        0.8,
    ),
    (
        "english_goal",
        re.compile(
            r"\b(speak(?:ing)? english|talk in english|english for (?:my )?"
            r"(?:job|work|career|studies|college|interview))\b",
            re.I,
        ),
        1,
        0.78,
    ),
    (
        "education",
        re.compile(
            r"\b(i(?:'m| am) (?:a )?student|i study|i(?:'m| am) studying|i graduated)\b"
            r"(?:\s+([a-z][a-z ]{2,30}))?",
            re.I,
        ),
        1,
        0.72,
    ),
    (
        "family",
        re.compile(
            r"\b(?:i live with|i stay with|i stay at home with)\s+(?:my\s+)?"
            r"(parents|family|wife|husband|mom|mother|dad|father|kids|son|daughter|"
            r"brother|sister|flatmates|roommates)\b",
            re.I,
        ),
        1,
        0.82,
    ),
    (
        "family",
        re.compile(
            r"\b(?:i have|i've got)\s+(?:a |an )?"
            r"(brother|sister|son|daughter|wife|husband|family|parents)\b",
            re.I,
        ),
        1,
        0.8,
    ),
    (
        "family",
        re.compile(
            r"\bmy (parents|family|wife|husband|mom|mother|dad|father|"
            r"brother|sister|kids|son|daughter)\b",
            re.I,
        ),
        1,
        0.74,
    ),
    (
        "routine",
        re.compile(
            r"\b(i (?:go|leave|reach) (?:to )?(?:the )?(?:office|work|college|school)"
            r"(?:\s+at\s+\d{1,2})?)\b",
            re.I,
        ),
        1,
        0.7,
    ),
    (
        "routine",
        re.compile(
            r"\b(i wake up(?: at \d{1,2}(?::\d{2})?)?)\b",
            re.I,
        ),
        1,
        0.78,
    ),
    (
        "routine",
        re.compile(
            r"\b(in the morning i [a-z ]{3,40})",
            re.I,
        ),
        1,
        0.72,
    ),
    (
        "hobby",
        re.compile(
            r"\b(?:i like|i love|i enjoy|enjoy)\s+(?:to\s+)?([a-z][a-z ]{2,40})",
            re.I,
        ),
        1,
        0.75,
    ),
    (
        "hobby",
        re.compile(
            r"\b(?:i(?:'m| am) playing|i play|playing)\s+([a-z][a-z ]{2,30})",
            re.I,
        ),
        1,
        0.84,
    ),
)

_NAME_STOP = frozenset(
    """
    not yes yeah ok okay good from very just like doing living working
    feeling excited confused sorry wait well then also about what how
    your name doing developer student really actually indian english
    use via us ami imi imius used using playing cricket football coding
    live laying software better ready fine work
    """.split()
)

_FILLER = frozenset("a an the to in on of for with and or".split())

# Words that carry no answer — safe to appear/disappear between STT partials
FILLER_TAIL = frozenset(
    """
    a an the and or but so yeah yes no um uh hmm oh like just very really too
    also is are am was were be been being do does did doing done have has had
    i you he she it we they me my your his her our their this that these those
    in on at of for to from with about by as well now then there here
    still already only even much many some any not dont don't cant can't
    ok okay actually basically mean means i'm you're it's that's thats
    """.split()
)

# Grammar scaffolding a correction may add on its own
GRAMMAR_WORDS = frozenset(
    """
    a an the and or but so is are am was were be been being do does did doing
    have has had will would can could should i you he she it we they me my your
    his her our their this that these those in on at of for to from with about
    by as not dont don't i'm you're it's say saying said very really just
    name called because then
    """.split()
)

_PLACE_STOP = frozenset(
    """
    before after here there now then that this those these where what which when
    yeah yes not just like living brother bhai thing way india english
    very good okay well also about
    """.split()
)

_JOB_STOP = frozenset(
    """
    good better nice fine great batsman player fan thing person guy
    man woman friend brother bhai coding ready
    """.split()
)

_PUSHBACK = re.compile(
    r"\b((?:do not|don't|dont|won'?t)\s+(?:answer|tell|say)|stop asking|"
    r"too many questions?|not that question|you asking me|question that i won'?t)\b",
    re.I,
)

_PAST_TIME = re.compile(r"\b(yesterday|last night|last week|last month|\d+ days? ago)\b", re.I)
_PAST_PRESENT = re.compile(
    r"\b(go|goes|eat|come|buy|see|do|get|take|play)\b",
    re.I,
)
_PAST_OK = re.compile(
    r"\b(went|ate|came|bought|saw|did|got|took|played|was|were)\b",
    re.I,
)
_AM_BASE = re.compile(r"\bi am (go|eat|come|buy|see|do|get|take)\b", re.I)

INTEREST_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gaming", ("gaming", "video game", "video games", "play games", "gta", "valorant", "pubg", "fortnite")),
    ("technology", ("coding", "programming", "software", "developer", "artificial intelligence")),
    ("cricket", ("cricket", "batsman", "bowler", "ipl")),
    ("business", ("startup", "business", "entrepreneur")),
    ("movies", ("movie", "movies", "film", "netflix", "series")),
    ("music", ("music", "song", "songs", "singing")),
    ("sports", ("football", "soccer", "gym", "fitness", "workout")),
    ("travel", ("travel", "travelling", "trip", "vacation")),
    ("food", ("cooking", "recipe", "recipes")),
    ("cars", ("car", "cars", "bike", "bikes")),
    ("content", ("youtube", "content creation", "instagram")),
)

_LOVE = re.compile(r"\b(i love|i really like|my favorite|my favourite)\b", re.I)
_TECH_AI = re.compile(r"\bai\b", re.I)

_CORR_CUE = re.compile(
    r"\b(no|not|wrong|correct|actually|already told|i mean|it's not|it is not)\b",
    re.I,
)
_FROM_PLACE = re.compile(
    r"\b(?:from|live in|living in|stay in|staying in)\s+([a-z]{3,})",
    re.I,
)
_NOT_FROM = re.compile(r"\bnot from\s+([a-z]{3,})", re.I)


@dataclass(frozen=True)
class Candidate:
    slot: str
    value: str
    confidence: float
    correction: bool = False
    rejected: str = ""


def _clean_value(raw: str) -> str:
    words = [w for w in clean_speech_text(raw).lower().split() if w not in _FILLER]
    if not words:
        return ""
    return " ".join(words[:4]).strip()


def _title(value: str) -> str:
    return " ".join(p.capitalize() for p in value.split() if p)


def is_plausible_name(value: str) -> bool:
    words = (value or "").lower().split()
    if not words:
        return False
    for w in words:
        if w in _NAME_STOP or w.endswith("ing") or len(w) < 3:
            return False
    return True


def is_plausible_place(value: str) -> bool:
    w = (value or "").strip().lower()
    if not w or " " in w:
        w = w.split()[0] if w else ""
    if len(w) < 4:
        return False
    if w in _PLACE_STOP or w in _NAME_STOP:
        return False
    return True


def is_plausible_job(value: str) -> bool:
    words = (value or "").strip().lower().split()
    if not words or words[0] in _JOB_STOP:
        return False
    if any(w in _JOB_STOP for w in words):
        return False
    return True


_HOBBY_STOP = frozenset(
    """
    lot when am is are was were be been being doing like just very really
    good fine nice that this those these thing things something
    """.split()
)


def is_plausible_hobby(value: str) -> bool:
    words = (value or "").strip().lower().split()
    if not words:
        return False
    content = [w for w in words if w not in _HOBBY_STOP and w not in _FILLER]
    if not content:
        return False
    if any(w in {"when", "whether"} for w in words):
        return False
    return True


_OK_SHORT = frozenset("yes no yeah yup ok okay sure right hi hello fine good".split())


def is_fragment(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    words = re.findall(r"[a-z']+", t.lower())
    if not words:
        return True
    if words[0] in _OK_SHORT and len(words) <= 3:
        return False
    if len(words) <= 2:
        return True
    if re.search(r"\b(the|a|an|to|in|my|and|or)\s*[.?!]?$", t, re.I):
        return True
    return False


def is_pushback(user_text: str) -> bool:
    return bool(_PUSHBACK.search(user_text or ""))


def detect_interests(user_text: str) -> list[tuple[str, str]]:
    """Observed interest labels with LOW/MEDIUM/HIGH — only from this line."""
    t = f" {(user_text or "").lower()} "
    if not t.strip():
        return []
    love = bool(_LOVE.search(t))
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, cues in INTEREST_CUES:
        if key in seen:
            continue
        if any(re.search(rf"\b{re.escape(cue)}\b", t) for cue in cues):
            seen.add(key)
            found.append((key, "HIGH" if love else "MEDIUM"))
    if "technology" not in seen and _TECH_AI.search(t):
        found.append(("technology", "HIGH" if love else "MEDIUM"))
    return found


def detect_errors(user_text: str) -> list[str]:
    """Conservative grammar tags. Empty unless the evidence is clear."""
    t = user_text or ""
    tags: list[str] = []
    if _PAST_TIME.search(t) and _PAST_PRESENT.search(t) and not _PAST_OK.search(t):
        tags.append("past tense")
    if _AM_BASE.search(t):
        tags.append("am + base verb")
    return tags


def is_correction_cue(user_text: str) -> bool:
    return bool(_CORR_CUE.search(user_text or ""))


def wants_recall(user_text: str) -> bool:
    t = user_text or ""
    if not _RECALL_ASK.search(t):
        return False
    # A recall needs an asking cue — "my name is ..." is a statement
    if not _ASK_CUE.search(t):
        return False
    if re.search(r"\bmy name(?:\s+is)?\s+[a-z]{3,}", t, re.I) and not re.search(
        r"\bdo you know\b", t, re.I
    ):
        return False
    return True


_Q_START = re.compile(
    r"^(what|who|where|when|why|how|do|did|does|can|could|are|is|have|tell)\b",
    re.I,
)


def is_question_sentence(part: str) -> bool:
    t = (part or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return bool(_Q_START.match(t))


def question_fingerprint(part: str) -> str:
    words = [w for w in re.findall(r"[a-z']+", (part or "").lower()) if w not in _FILLER]
    return " ".join(words[:8])


def question_lines(coach_text: str) -> list[str]:
    """Short fingerprints of the questions the coach just asked."""
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", coach_text or ""):
        part = part.strip()
        if not is_question_sentence(part):
            continue
        key = question_fingerprint(part)
        if key:
            out.append(key)
    return out


def quoted_target(coach_text: str) -> str:
    """The model sentence a learner was asked to repeat, if any."""
    if not _TRY_CUE.search(coach_text or ""):
        return ""
    m = re.search(
        r"say it like this:\s*[\"“']([A-Za-z][^\"”']{2,80})[\"”']",
        coach_text or "",
        re.I,
    )
    if m and len(m.group(1).split()) >= 2:
        return m.group(1).strip()
    found: list[str] = []
    for m in re.finditer(r"[\"“']([A-Za-z][^\"”']{2,80})[\"”']", coach_text or ""):
        phrase = m.group(1).strip()
        if len(phrase.split()) >= 2:
            found.append(phrase)
    return found[-1] if found else ""


def fix_is_faithful(target: str, user_text: str) -> bool:
    """A fix may reshape grammar, never swap what the learner talked about."""
    said = set(re.findall(r"[a-z']+", (user_text or "").lower()))
    words = [w for w in re.findall(r"[a-z']+", (target or "").lower())]
    content = [w for w in words if w not in GRAMMAR_WORDS]
    if not content:
        return False
    for w in content:
        if w in said:
            continue
        # allow word-form changes: doing -> do, batting -> bat
        stem = w[:4]
        if len(w) >= 4 and any(s.startswith(stem) for s in said):
            continue
        return False
    return True


def strip_unfaithful_fix(coach_text: str, user_text: str) -> str:
    """Drop a correction that invented content the learner never said."""
    target = quoted_target(coach_text)
    if not target or fix_is_faithful(target, user_text):
        return coach_text
    keep: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", coach_text or ""):
        low = part.lower()
        if target.lower()[:20] in low or _TRY_CUE.search(low) or '"' in part:
            continue
        keep.append(part.strip())
    out = " ".join(p for p in keep if p).strip()
    return out or "Sorry, I missed that. Can you say the whole sentence again?"


def phrase_match(said: str, target: str) -> bool:
    """Did the learner actually repeat the corrected sentence?"""
    want = {w for w in re.findall(r"[a-z']+", (target or "").lower()) if w not in _FILLER}
    got = {w for w in re.findall(r"[a-z']+", (said or "").lower()) if w not in _FILLER}
    if not want:
        return False
    return len(want & got) / len(want) >= 0.65


def name_only_ask(user_text: str) -> bool:
    t = user_text or ""
    return bool(_NAME_ONLY_ASK.search(t)) and not re.search(
        r"\babout me\b", t, re.I
    )


def extract_candidates(user_text: str) -> list[Candidate]:
    """Ranked fact candidates from one final utterance."""
    text = clean_speech_text(user_text or "")
    if not text:
        return []
    found: list[Candidate] = []

    m = _CORR_NOT_THEN_NEW.search(text)
    if m:
        old, new = _clean_value(m.group(1)), _clean_value(m.group(2))
        if new and new.split()[0] not in _NAME_STOP:
            found.append(
                Candidate("name", _title(new), 0.96, correction=True, rejected=_title(old))
            )
    m = _CORR_NEW_NOT_OLD.search(text)
    if m:
        new, old = _clean_value(m.group(1)), _clean_value(m.group(2))
        if new and is_plausible_name(new):
            found.append(
                Candidate("name", _title(new), 0.96, correction=True, rejected=_title(old))
            )
    m = _CORR_NO_I_AM.search(text)
    if m:
        new = _clean_value(m.group(1))
        if new and is_plausible_name(new):
            found.append(
                Candidate("name", _title(new), 0.96, correction=True, rejected="")
            )

    for slot, pat, group, conf in _SLOT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if group == 0:
            parts = [g for g in m.groups() if g]
            raw = " ".join(parts[:2])
        else:
            raw = m.group(group) or ""
        value = _clean_value(raw)
        if slot == "education" and not value:
            value = "student"
        if slot == "education" and value.lower() in {
            "i'm student",
            "i am student",
            "i study",
            "i'm studying",
            "i am studying",
            "i graduated",
        }:
            extra = ""
            if m.lastindex and m.lastindex >= 2:
                extra = _clean_value(m.group(2) or "")
            value = extra or "student"
        if not value:
            continue
        if slot == "english_goal" and "english" not in text.lower():
            continue
        if slot == "job" and not is_plausible_job(value):
            continue
        if slot == "english_goal":
            value = " ".join(value.split()[:6])
        if slot == "name":
            if re.search(r"\b(?:who am i|am i right|am i correct)\b", text, re.I):
                continue
            if not is_plausible_name(value):
                continue
            if conf < 0.85 and len(value.split()[0]) < 4:
                continue
            # First+last looks like a real name — strong enough to replace garbage
            if len(value.split()) >= 2:
                conf = max(conf, 0.91)
        if slot == "hobby":
            value = value.split(".")[0].strip()
            if not value or not is_plausible_hobby(value):
                continue
        if slot in {"native", "current_city"} and not is_plausible_place(value):
            continue
        found.append(Candidate(slot, _title(value) if slot != "job" else value, conf))

    # Place corrections: "not from Before, I am from Bhopal"
    places = [_clean_value(m.group(1)) for m in _FROM_PLACE.finditer(text)]
    places = [p for p in places if is_plausible_place(p)]
    if places and is_correction_cue(text):
        winner = _title(places[-1])
        found.append(
            Candidate("current_city", winner, 0.96, correction=True, rejected="")
        )

    found.sort(key=lambda c: (c.correction, c.confidence), reverse=True)
    return found
