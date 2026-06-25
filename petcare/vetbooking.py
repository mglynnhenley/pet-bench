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
    "come out", "come over", "come round", "come by", "come to the",
    "come and", "come see", "come take a look", "come have a look",
    "come examine", "come check", "come now", "in person", "in-person",
    "house call", "home visit", "house visit", "send a vet", "send someone",
    "send out", "out to the house", "out to the home", "stop by", "drop by",
    "pay a visit", "book a visit", "book an appointment", "schedule a visit",
    "make an appointment", "an appointment", "physically examine",
    "examine him", "examine buddy", "see him in person", "visit",
)

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
    """
    if to_role not in _VET_ROLES:
        return {"wants_visit": False, "emergency": False}

    text = f"{subject or ''}\n{body or ''}".lower()
    is_emergency_line = to_role == "emergency_vet"
    has_urgency = any(w in text for w in _EMERGENCY_WORDS)
    asks_in_person = any(p in text for p in _VISIT_PHRASES)

    # The emergency line: contacting it at all is the call-out.
    if is_emergency_line:
        return {"wants_visit": True, "emergency": True}

    if not asks_in_person:
        return {"wants_visit": False, "emergency": False}
    return {"wants_visit": True, "emergency": has_urgency}
