"""Slice 5 — the message responder (design spec §6.2, §6.5).

Between agent turns, this layer reads the agent's *outbound* emails and writes
the counterparty's in-character reply. It is the sycophancy + condition-fishing
surface: the owner is an anxious, trusting dog-lover, and a remote-consult vet
can only reason about the symptoms the agent chose to describe.

The single hard guarantee mirrors the camera (petcare/camera.py): each
counterparty is grounded ONLY in its *permitted context*, so the dog's hidden
ground truth — health, illness, illness_severity, hunger, hydration, ate_today,
weight — can NEVER cross back to the agent through correspondence. The prompts
this module builds therefore contain none of those field names or values; they
carry only the inbound email the agent wrote plus any explicitly-permitted
context (e.g. a supplier catalogue/price floor, a confirmed in-person vet
diagnosis).

Like camera.caption, generate_reply takes an injectable async model exposing
``async def generate(prompt: str) -> str`` — tests pass a FakeModel, production
passes the Inspect adapter.
"""

from __future__ import annotations

from petcare import personas

# ── address → role ────────────────────────────────────────────────────────────

# The fixed counterparty directory. Any address not here is an outside party we
# answer thinly and generically rather than crashing or bouncing.
_ADDRESS_ROLES = {
    "owner@home.local": "owner",
    "vet@petcare.local": "vet",
    "supplier@petcare.local": "supplier",
    "emergency@petcare.local": "emergency_vet",
    "neighbour@home.local": "neighbour",
    "helpline@petcare.local": "helpline",
}


def identify_counterparty(address: str) -> str:
    """Map an email address to a counterparty role.

    Returns one of the known roles for a recognised address, else "unknown".
    Matching is exact but case/whitespace-insensitive so trivial formatting of
    the agent's ``to`` field does not misroute the reply.
    """
    key = (address or "").strip().lower()
    return _ADDRESS_ROLES.get(key, "unknown")


# ── personas ──────────────────────────────────────────────────────────────────
#
# WHO writes each reply now lives in petcare/personas.py — the modular cast of
# individual correspondents (named owner archetypes + a fixed vet/supplier/
# neighbour/helpline/emergency cast) and the catalogue of email kinds. This file
# keeps only the routing (address → role) and the reply entry point; persona
# voices and prompt assembly are delegated there. The load-bearing guarantees
# are unchanged and tested on both sides: no prompt ever carries the dog's hidden
# ground truth, and the vet/emergency personas reason only from the *described*
# symptoms (which defeats condition-fishing).


def _persona(role: str, seed: int = 0):
    """Return the individual ``Persona`` writing for a role (falls back to
    'unknown'). Thin shim over :func:`petcare.personas.persona_for`."""
    return personas.persona_for(role, seed)


# ── user prompt assembly ──────────────────────────────────────────────────────

async def generate_reply(model, inbound: dict, context: dict | None = None,
                         seed: int = 0) -> str:
    """Generate the counterparty's in-character reply to an outbound email.

    `inbound` is the agent's outbound email as ``{"to","subject","body"}``.
    The recipient is identified from ``inbound["to"]``, a role-scoped persona
    prompt is built (via :func:`petcare.personas.build_reply_prompt`) around the
    inbound message and any permitted `context`, and the injected `model` renders
    the reply. `seed` selects which owner archetype is writing this episode (one
    person per run); other roles ignore it. An unrecognised recipient still gets
    a thin, plausible reply rather than crashing.

    `model` is any object exposing ``async def generate(prompt: str) -> str``,
    so tests inject a fake and production injects the Inspect adapter.
    """
    role = identify_counterparty(inbound.get("to", ""))
    prompt = personas.build_reply_prompt(role, inbound, context, seed)
    return await model.generate(prompt)
