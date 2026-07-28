"""Counterparty decisions: the persona IS the actuator (pure, no Inspect dep).

Slice 19. Until now a counterparty's *words* were written by an LLM holding the
whole email thread, while its *actions* were decided by a memoryless keyword
regex (`vetbooking.detect_visit_request`). That split authority made the
environment say one thing and do another — audited on a real eval log:

  * "I'd like to schedule a consultation"        → booked a £200 emergency call-out
  * "Just confirming the scheduled vet visit"    → booked a SECOND £80 visit
  * "Can we arrange emergency intervention?"     → booked nothing at all

A high-fidelity twin needs one mind per counterparty deciding both. So the reply
prompt now asks the persona to append a machine-readable DECISION line, and the
world acts on *that*. This module is the pure protocol: split the control line
off the prose (the agent must never see it) and parse it.

A missing or unparseable line yields ``None`` rather than ``False`` — "the model
didn't answer" is not "the model declined", and only the former should fall back
to the legacy heuristic.
"""

from __future__ import annotations

import re

# The control line the persona appends. Matched case-insensitively, anywhere in
# the completion, so a chatty model that adds a sign-off after it still parses.
_DECISION_RE = re.compile(r"^[ \t>*_-]*decision\s*:\s*(?P<body>.*)$",
                          re.IGNORECASE | re.MULTILINE)

_TRUE = {"yes", "y", "true", "t", "1"}
_FALSE = {"no", "n", "false", "f", "0", "none"}

# Items the world can actually fulfil (see World.place_order / Config.catalogue).
# The persona is held to this list so it can't invent a SKU the world will drop.
_ORDERABLE = {"dog_food", "med_gi_relief"}
_MAX_QTY = 20


def split_decision(text: str) -> tuple[str, str | None]:
    """Split a completion into ``(reply_prose, raw_decision)``.

    The decision line is REMOVED from the prose: it is a control channel between
    the persona and the world, and letting it reach the agent would teach it the
    harness's internals. ``raw_decision`` is None when no line is present.
    """
    text = text or ""
    m = _DECISION_RE.search(text)
    if m is None:
        return text, None
    prose = (text[:m.start()] + text[m.end():]).strip()
    return prose, m.group("body").strip()


def _fields(raw: str | None) -> dict:
    """Parse ``key=value`` pairs from a decision body, tolerating `,` / `;` /
    whitespace separators and spaces around `=` (small models are loose)."""
    if not raw:
        return {}
    out = {}
    for key, value in re.findall(r"(\w+)\s*=\s*([^,;\s]+)", raw):
        out[key.lower()] = value.strip().lower()
    return out


def _flag(fields: dict, key: str) -> bool | None:
    """A tri-state boolean: True / False / None when absent or unrecognised."""
    v = fields.get(key)
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def parse_vet_decision(text: str) -> dict:
    """Parse a vet/emergency-vet completion.

    Returns ``{"reply", "books_visit", "emergency"}``. ``books_visit`` is None
    when the persona gave no usable decision, so the caller can fall back to
    :func:`petcare.vetbooking.detect_visit_request` instead of silently doing
    nothing. An emergency always implies a dispatch — an emergency call-out that
    sends nobody is incoherent.
    """
    prose, raw = split_decision(text)
    fields = _fields(raw)
    emergency = _flag(fields, "emergency") is True
    # `dispatch` is the field the prompt asks for: "visit" proved ambiguous in live
    # testing — asked to confirm an already-arranged appointment, the model answered
    # visit=yes meaning "yes, a visit is happening", which bought a SECOND call-out.
    # "visit" is still accepted as a fallback for robustness.
    books = _flag(fields, "dispatch")
    if books is None:
        books = _flag(fields, "visit")
    if emergency:
        books = True
    return {"reply": prose, "books_visit": books, "emergency": emergency}


def parse_supplier_decision(text: str) -> dict:
    """Parse a supplier completion.

    Returns ``{"reply", "places_order", "item", "qty", "expedited"}``.
    ``places_order`` is None when no decision was given. An order for an item the
    world cannot fulfil is refused (False) rather than silently dropped, and the
    quantity is clamped so a hallucinated figure can't drain the care budget.
    """
    prose, raw = split_decision(text)
    fields = _fields(raw)
    places = _flag(fields, "order")
    item = fields.get("item")
    qty_raw = fields.get("qty", "1")

    if places:
        if item not in _ORDERABLE:
            places = False           # can't fulfil what isn't in the catalogue
        else:
            try:
                qty = max(1, min(int(re.sub(r"\D", "", qty_raw) or 1), _MAX_QTY))
            except ValueError:
                qty = 1
            return {"reply": prose, "places_order": True, "item": item,
                    "qty": qty, "expedited": _flag(fields, "expedited") is True}
    return {"reply": prose, "places_order": places, "item": None,
            "qty": 0, "expedited": False}
