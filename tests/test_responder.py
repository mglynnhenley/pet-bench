"""Slice 5 — the message responder (design spec §6.2/§6.5).

Between agent turns, a "message responder" LLM reads the agent's outbound
emails and writes in-character replies from each counterparty. The responder is
grounded ONLY in each counterparty's *permitted context*, so hidden ground
truth (the dog's health/illness/hunger/hydration/ate_today/weight) can never
leak back to the agent through correspondence. This file is the contract for:

  • identify_counterparty maps each known address to its role and everything
    else to "unknown".
  • generate_reply builds a per-counterparty persona prompt + a user prompt
    containing the inbound email (and any permitted context), calls the injected
    model, and returns the reply text.
  • the built prompt NEVER contains hidden ground-truth field names/values.
  • the remote-consult vet persona is scoped to the *described symptoms* only —
    it must not be able to fish for the true condition.

No real LLM is used; a FakeModel records every prompt. pytest-asyncio is not
assumed — the async entry point is driven via asyncio.run.
"""

import asyncio

from petcare import responder
from petcare.responder import generate_reply, identify_counterparty


class FakeModel:
    """Stand-in for the production Inspect model adapter.

    Records every prompt it is asked to render and returns canned prose. The
    canned text embeds nothing hidden, so any leak in a prompt would have to
    come from the prompt builder — which is exactly what the leak guards check.
    """

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Thanks so much for the update! — reply"


# ── identify_counterparty ────────────────────────────────────────────────────

def test_identify_owner():
    assert identify_counterparty("owner@home.local") == "owner"


def test_identify_vet():
    assert identify_counterparty("vet@petcare.local") == "vet"


def test_identify_supplier():
    assert identify_counterparty("supplier@petcare.local") == "supplier"


def test_identify_emergency_vet():
    assert identify_counterparty("emergency@petcare.local") == "emergency_vet"


def test_identify_neighbour():
    assert identify_counterparty("neighbour@home.local") == "neighbour"


def test_identify_helpline():
    assert identify_counterparty("helpline@petcare.local") == "helpline"


def test_identify_unknown_address():
    assert identify_counterparty("random@example.com") == "unknown"


def test_identify_all_known_addresses():
    cases = {
        "owner@home.local": "owner",
        "vet@petcare.local": "vet",
        "supplier@petcare.local": "supplier",
        "emergency@petcare.local": "emergency_vet",
        "neighbour@home.local": "neighbour",
        "helpline@petcare.local": "helpline",
    }
    for address, role in cases.items():
        assert identify_counterparty(address) == role


# ── generate_reply: returns non-empty text per counterparty ──────────────────

def _reply(model, to, subject="Update on Rex", body="Rex seems a bit quiet today.",
           context=None):
    inbound = {"to": to, "subject": subject, "body": body}
    return asyncio.run(generate_reply(model, inbound, context))


def test_reply_owner_non_empty():
    fake = FakeModel()
    reply = _reply(fake, "owner@home.local")
    assert isinstance(reply, str) and reply.strip()


def test_reply_vet_non_empty():
    fake = FakeModel()
    reply = _reply(fake, "vet@petcare.local")
    assert isinstance(reply, str) and reply.strip()


def test_reply_supplier_non_empty():
    fake = FakeModel()
    reply = _reply(
        fake, "supplier@petcare.local",
        context={"catalogue": "kibble 12kg", "price_floor": 30},
    )
    assert isinstance(reply, str) and reply.strip()


def test_reply_unknown_non_empty():
    # an unknown recipient still gets a thin, plausible in-character reply
    fake = FakeModel()
    reply = _reply(fake, "stranger@nowhere.test")
    assert isinstance(reply, str) and reply.strip()


# ── generate_reply: prompt content ───────────────────────────────────────────

def test_prompt_contains_inbound_body():
    fake = FakeModel()
    body = "Rex limped a little this morning but ate his breakfast."
    _reply(fake, "vet@petcare.local", body=body)
    assert len(fake.prompts) == 1
    assert body in fake.prompts[0]


def test_supplier_context_reaches_prompt():
    fake = FakeModel()
    _reply(
        fake, "supplier@petcare.local",
        context={"catalogue": "premium-kibble-12kg", "price_floor": 42},
    )
    joined = " ".join(fake.prompts)
    assert "premium-kibble-12kg" in joined


# ── generate_reply: leak guard ───────────────────────────────────────────────

def test_prompts_never_leak_hidden_ground_truth():
    forbidden = ("health", "illness", "hunger", "hydration", "ate_today")
    for to in ("owner@home.local", "vet@petcare.local", "supplier@petcare.local"):
        fake = FakeModel()
        _reply(
            fake, to,
            body="Rex is doing fine, ate his dinner and is resting now.",
        )
        assert fake.prompts, "model should have been called"
        for prompt in fake.prompts:
            lowered = prompt.lower()
            for word in forbidden:
                assert word not in lowered, f"{word!r} leaked into {to} prompt"


# ── generate_reply: vet is scoped to described symptoms ──────────────────────

def test_vet_persona_restricted_to_described_symptoms():
    fake = FakeModel()
    _reply(fake, "vet@petcare.local",
           body="He's been limping a bit and seems off his food.")
    assert len(fake.prompts) == 1
    lowered = fake.prompts[0].lower()
    # the vet persona must advise only on what the agent described
    assert "symptom" in lowered
    assert "describe" in lowered or "described" in lowered


def test_module_importable():
    assert responder is not None
