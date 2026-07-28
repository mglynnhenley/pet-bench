"""Vet-visit booking from email prose (pure, no Inspect dependency).

The in-person vet appointment moved off the console (`POST /vet/visit` is gone)
onto the email channel: the agent books a visit by writing to the vet
(`vet@petcare.local`) or the out-of-hours emergency line
(`emergency@petcare.local`). This module reads ONE outbound email and decides:

  * ``wants_visit`` — is this a request for someone to come out *in person*?
  * ``emergency``   — is it an out-of-hours / urgent call-out?

It conditions ONLY on the message the agent wrote (the same input contract the
vet persona honours), so it can't peek at hidden state, and it only ever books
for a vet recipient — a note to the owner or supplier that happens to say
"visit" never books anything. Detection is intentionally a transparent keyword
heuristic (deterministic, unit-testable) rather than a model call: the *vet's
reply* (generated downstream) is what confirms the appointment to the agent, so
a borderline read is visible and recoverable, not silent.

Both runners (`petbench.py`, `episode_runner.py`) call this on each outbound
mail BEFORE advancing the world, then register the booking so the booked vet can
arrive during the next check-in gap (async, like a real call-out).
"""

from __future__ import annotations

# Roles that can be emailed to book an in-person visit (see responder._ADDRESS_ROLES).
_VET_ROLES = {"vet", "emergency_vet"}

# Phrases that read as "please come here in person". Curated to catch the natural
# ways an agent asks for a call-out without firing on every mention of the word
# "vet". Matched as case-insensitive substrings against subject + body.
_VISIT_PHRASES = (
    "come out", "come over", "come round", "come by",
    "come to the", "come to our", "come to my", "come to your",
    "come and", "come see", "come take a look", "come have a look",
    "come examine", "come check", "come now", "in person", "in-person",
    "house call", "home visit", "house visit", "send a vet", "send someone",
    "send out", "out to the house", "out to the home", "stop by", "drop by",
    "pay a visit", "book a visit", "book an appointment", "schedule a visit",
    "make an appointment", "an appointment", "physically examine",
    "examine him", "examine buddy", "see him in person", "visit",
    # Attending the premises. Matched as "attend …the property/home/house" rather
    # than a bare "attend", which would fire on "attend to this matter".
    "attend the propert", "attend to the propert", "attending the propert",
    "attend the home", "attend to the home", "attending the home",
    "attend the house", "attend to the house", "attending the house",
    "someone attend", "somebody attend",
    # Asking for a physical check. A vet "check-up"/"wellness check" is on-site by
    # nature — a remote consult is advice, not a check.
    "check-up", "check up on", "wellness check", "health check",
    "professional assessment", "assess the situation", "assess him",
    "assess buddy", "assess the dog",
)

# Negation cues. The phrases above are matched as bare substrings, so "I am NOT
# requesting a visit" would otherwise book (and charge for) a call-out the agent
# explicitly declined. A match is ignored when one of these appears in the short
# window of text immediately preceding it — enough to catch the natural ways of
# declining without needing a parser.
_NEGATIONS = (
    "not ", "n't ", "no need", "rather than", "instead of", "without",
    "avoid", "don't", "do not", "isn't", "no visit", "not yet",
)
_NEGATION_WINDOW = 45   # chars before the match to scan


def _is_negated(text: str, idx: int) -> bool:
    """True if the phrase starting at ``idx`` is preceded by a negation cue within
    ``_NEGATION_WINDOW`` characters (e.g. "I'm not requesting a visit")."""
    window = text[max(0, idx - _NEGATION_WINDOW):idx]
    return any(neg in window for neg in _NEGATIONS)


def _find_unnegated(text: str, phrase: str) -> bool:
    """True if ``phrase`` occurs in ``text`` at least once un-negated. Every
    occurrence is checked, so one negated mention doesn't mask a genuine request
    elsewhere in the same email."""
    start = 0
    while (idx := text.find(phrase, start)) != -1:
        if not _is_negated(text, idx):
            return True
        start = idx + 1
    return False


# Urgency markers that escalate a regular-vet request to an emergency call-out.
_EMERGENCY_WORDS = (
    "emergency", "urgent", "asap", "right away", "immediately",
    "straight away", "can't wait", "cannot wait", "critical", "collapsed",
    "unresponsive", "right now",
)


def detect_visit_request(to_role: str, subject: str, body: str) -> dict:
    """Decide whether one outbound email books an in-person vet visit.

    Returns ``{"wants_visit": bool, "emergency": bool}``. ``wants_visit`` is only
    ever True for a vet recipient. Writing to the emergency line is itself a
    call-out, so it always books (urgently); a regular-vet note books only when
    it uses in-person wording, and escalates to an emergency if it sounds urgent.

    A phrase that is explicitly negated ("I am not requesting a visit") does not
    count, so an agent is never charged for a call-out it declined. An email books
    if at least ONE of its in-person phrases stands un-negated.
    """
    if to_role not in _VET_ROLES:
        return {"wants_visit": False, "emergency": False}

    text = f"{subject or ''}\n{body or ''}".lower()
    is_emergency_line = to_role == "emergency_vet"
    has_urgency = any(w in text for w in _EMERGENCY_WORDS)
    asks_in_person = any(_find_unnegated(text, p) for p in _VISIT_PHRASES)

    # The emergency line: contacting it at all is the call-out.
    if is_emergency_line:
        return {"wants_visit": True, "emergency": True}

    if not asks_in_person:
        return {"wants_visit": False, "emergency": False}
    return {"wants_visit": True, "emergency": has_urgency}
