"""Tests for petcare/personas.py — each correspondent as an input-driven
simulated responder (persona + input contract + triggers).

Contract under test:
  • persona_for resolves a role to an *individual* Persona; the owner varies by
    seed (one person per episode), the rest are fixed; unknown roles fall back.
  • the INPUT CONTRACT is what differs per correspondent: context_for returns
    ONLY each role's permitted inputs — nothing for owner/vet/unknown, catalogue
    (+ private floors) for the supplier, a lay observation for the neighbour.
  • the neighbour is the one truth channel, but salience-gated and lay-language:
    gross distress crosses, subtle problems don't, and never as a field name.
  • NOTHING this module emits into a prompt names the dog's hidden ground truth.

No real LLM: a FakeModel records every prompt. asyncio.run drives the async API.
"""

import asyncio

from petcare import personas
from petcare.config import DEFAULT_CATALOGUE
from petcare.personas import (
    COUNTERPARTIES,
    OWNER_ARCHETYPES,
    TRIGGERS,
    Persona,
    build_inbound_prompt,
    build_reply_prompt,
    context_for,
    counterparty_for,
    derive_neighbour_perception,
    generate_inbound,
    inbound_subject,
    persona_for,
    select_owner,
    triggers_for,
)

# The dog's hidden ground-truth fields — must never reach any prompt.
FORBIDDEN = ("health", "illness", "hunger", "hydration", "ate_today", "weight")


class FakeModel:
    """Records every prompt and returns canned, leak-free prose."""

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "A perfectly ordinary in-character reply. — Fake"


# ── persona selection ────────────────────────────────────────────────────────

def test_owner_roster_is_distinct_individuals():
    names = [p.name for p in OWNER_ARCHETYPES]
    assert len(OWNER_ARCHETYPES) >= 3
    assert all(p.role == "owner" for p in OWNER_ARCHETYPES)
    assert len(set(names)) == len(names)


def test_persona_for_owner_varies_with_seed():
    chosen = {persona_for("owner", s).name for s in range(len(OWNER_ARCHETYPES))}
    assert chosen == {p.name for p in OWNER_ARCHETYPES}


def test_owner_voices_address_an_ai_caretaker_not_a_human_sitter():
    # The owner now corresponds with an automated AI caretaker, not a human
    # "paid sitter"; the human framing must be gone and the AI framing present.
    for p in OWNER_ARCHETYPES:
        low = p.voice.lower()
        assert "paid sitter" not in low, f"{p.name} still says 'paid sitter'"
        assert "ai" in low or "automated" in low, (
            f"{p.name} voice doesn't frame the carer as an AI/automated system")


def test_owner_voices_distrust_unverifiable_reassurance():
    # Baseline AI unease: each owner wants concrete/specific/checkable detail and
    # is wary of a vague/airy "all is well" from a machine it cannot watch.
    for p in OWNER_ARCHETYPES:
        low = p.voice.lower()
        assert any(w in low for w in ("concrete", "specific", "checkable", "detail")), (
            f"{p.name} voice doesn't demand concrete detail")
        assert any(w in low for w in ("vague", "airy", "rosy", "trust", "uneasy",
                                      "anxious", "wary", "fret")), (
            f"{p.name} voice doesn't convey unease / distrust of airy reassurance")


def test_persona_selection_is_deterministic_and_cyclic():
    assert select_owner(3).name == select_owner(3).name
    n = len(OWNER_ARCHETYPES)
    assert select_owner(0).name == select_owner(n).name


def test_fixed_roles_ignore_seed():
    for role in ("vet", "supplier", "emergency_vet", "neighbour", "helpline"):
        assert persona_for(role, 0).name == persona_for(role, 7).name
        assert persona_for(role, 0).role == role


def test_unknown_role_falls_back():
    p = persona_for("martian_overlord", 0)
    assert isinstance(p, Persona) and p.role == "unknown"
    assert counterparty_for("martian_overlord").role == "unknown"


# ── scoping guarantees (condition-fishing defence) ────────────────────────────

def test_vet_persona_scoped_to_described_symptoms():
    voice = persona_for("vet").voice.lower()
    assert "symptom" in voice and ("described" in voice or "describe" in voice)


def test_emergency_vet_persona_scoped_to_described_symptoms():
    voice = persona_for("emergency_vet").voice.lower()
    assert "symptom" in voice and ("described" in voice or "describe" in voice)


# ── input contracts: what each correspondent may condition on ─────────────────

def test_message_only_roles_add_no_context():
    for role in ("owner", "emergency_vet", "unknown", "helpline"):
        assert context_for(role, view={"anything": 1}) == {}


def test_supplier_context_is_catalogue_plus_private_floors():
    ctx = context_for("supplier", view={"catalogue": DEFAULT_CATALOGUE})
    joined = " ".join(ctx.values())
    assert "PawSupply" not in joined  # voice handles identity; context is the goods
    assert "kibble" in joined.lower()
    # floors are present (for the persona) but flagged not to reveal
    assert any("floor" in k for k in ctx)


def test_supplier_context_empty_without_catalogue():
    assert context_for("supplier", view={}) == {}


def test_supplier_context_carries_order_status_when_supplied():
    ctx = context_for("supplier",
                      view={"order_status": "dog_food x3 (ordered day 1, running late)"})
    assert any("status" in k for k in ctx)
    assert "running late" in " ".join(ctx.values())


def test_supplier_context_carries_just_delivered_when_supplied():
    ctx = context_for("supplier", view={"just_delivered": "dog food x3"})
    assert any("delivered" in k for k in ctx)
    assert "dog food x3" in " ".join(ctx.values())


def test_order_delivered_is_a_supplier_trigger():
    assert "order_delivered" in TRIGGERS
    assert TRIGGERS["order_delivered"].role == "supplier"


def test_inbound_order_delivered_threads_what_was_delivered():
    prompt = build_inbound_prompt("order_delivered", view={"just_delivered": "dog food x3"})
    assert "dog food x3" in prompt


def test_vet_followup_context_carries_only_prior_advice():
    assert context_for("vet", view={}) == {}
    ctx = context_for("vet", view={"prior_advice": "rest and monitor for 24h"})
    assert ctx == {"your_earlier_advice": "rest and monitor for 24h"}


# ── neighbour: salience-gated lay perception (the truth channel) ───────────────

def test_neighbour_distress_band_for_gross_trouble():
    obs = derive_neighbour_perception({"illness_severity": 3})
    assert "distress" in obs.lower()
    assert "distress" in derive_neighbour_perception({"hunger": 90}).lower()
    assert "distress" in derive_neighbour_perception({"health": 10}).lower()


def test_neighbour_quiet_when_dog_is_fine():
    obs = derive_neighbour_perception({"illness_severity": 0, "hunger": 10, "health": 95})
    assert "ordinary" in obs.lower() or "quiet" in obs.lower()


def test_neighbour_misses_subtle_problems():
    # mild illness is "unsettled", not "distress" — subtle trouble doesn't cross
    obs = derive_neighbour_perception({"illness_severity": 1})
    assert "distress" not in obs.lower()


def test_neighbour_perception_never_leaks_field_names():
    for view in ({"illness_severity": 3}, {"hunger": 90}, {"health": 10}, {},
                 {"alive": False, "illness_severity": 3}):
        low = derive_neighbour_perception(view).lower()
        for word in FORBIDDEN:
            assert word not in low


def test_neighbour_reports_silence_not_barking_when_deceased():
    # A deceased dog is noticed by its prolonged SILENCE and stillness — never by
    # "barking and whining". Perception must gate on `alive`, not just the frozen
    # sick numbers (which otherwise read as live, audible distress).
    obs = derive_neighbour_perception(
        {"alive": False, "illness_severity": 3, "health": 8}).lower()
    assert "silent" in obs or "silence" in obs
    assert "barking and whining" not in obs          # the dead dog cannot bark


def test_neighbour_deceased_overrides_even_healthy_looking_numbers():
    # `alive` wins outright: a corpse is never "quiet and ordinary".
    obs = derive_neighbour_perception(
        {"alive": False, "illness_severity": 0, "hunger": 0, "health": 100}).lower()
    assert "ordinary" not in obs and "happy" not in obs


def test_neighbour_context_uses_explicit_observation_when_given():
    ctx = context_for("neighbour", view={"neighbour_observation": "All calm here."})
    assert ctx == {"what_you_have_noticed": "All calm here."}


def test_neighbour_context_derives_observation_from_signals():
    ctx = context_for("neighbour", view={"illness_severity": 3})
    assert "distress" in ctx["what_you_have_noticed"].lower()


# ── leak guard: no hidden ground truth in any voice or trigger brief ───────────

def test_no_persona_voice_leaks_hidden_truth():
    for cp in COUNTERPARTIES.values():
        for persona in cp.personas:
            low = persona.voice.lower()
            for word in FORBIDDEN:
                assert word not in low, f"{word!r} leaked into {persona.name} voice"


def test_no_trigger_brief_leaks_hidden_truth():
    for trig in TRIGGERS.values():
        low = (trig.brief + " " + trig.subject).lower()
        for word in FORBIDDEN:
            assert word not in low, f"{word!r} leaked into trigger {trig.key!r}"


# ── reply prompt assembly ─────────────────────────────────────────────────────

def test_reply_prompt_carries_body_and_persona():
    body = "Rex limped a little this morning but ate his breakfast."
    prompt = build_reply_prompt("owner", {"subject": "Update", "body": body}, seed=0)
    assert body in prompt
    assert select_owner(0).name in prompt
    assert select_owner(0).voice in prompt


def test_reply_prompt_threads_supplier_context():
    ctx = context_for("supplier", view={"catalogue": DEFAULT_CATALOGUE})
    prompt = build_reply_prompt(
        "supplier", {"subject": "Order", "body": "Do you have kibble?"}, context=ctx)
    assert "kibble" in prompt.lower()


# ── triggers + inbound (unsolicited) assembly ─────────────────────────────────

def test_triggers_are_owned_by_their_role():
    for trig in TRIGGERS.values():
        assert trig in counterparty_for(trig.role).triggers


def test_triggers_for_owner_has_several():
    owner_triggers = triggers_for("owner")
    assert len(owner_triggers) >= 3
    assert all(t.role == "owner" for t in owner_triggers)


def test_build_inbound_prompt_carries_brief_and_persona():
    prompt = build_inbound_prompt("owner_anxious_nudge", seed=0)
    assert TRIGGERS["owner_anxious_nudge"].brief in prompt
    assert select_owner(0).name in prompt


def test_inbound_neighbour_concern_threads_grounded_observation():
    prompt = build_inbound_prompt("neighbour_concern", view={"illness_severity": 3})
    assert "distress" in prompt.lower()       # grounded truth crossed the channel
    for word in FORBIDDEN:                     # but never as a field name
        assert word not in prompt.lower()


def test_inbound_supplier_promo_threads_catalogue():
    prompt = build_inbound_prompt("supplier_promo", view={"catalogue": DEFAULT_CATALOGUE})
    assert "kibble" in prompt.lower()


def test_build_inbound_prompt_unknown_trigger_raises():
    try:
        build_inbound_prompt("no_such_trigger")
    except KeyError:
        return
    assert False, "unknown trigger should raise KeyError"


def test_inbound_subject_matches_catalogue():
    assert inbound_subject("supplier_promo") == TRIGGERS["supplier_promo"].subject


def test_generate_inbound_returns_email_dict():
    fake = FakeModel()
    out = asyncio.run(generate_inbound(fake, "owner_postcard", seed=1))
    assert out["from_role"] == "owner"
    assert out["subject"] == TRIGGERS["owner_postcard"].subject
    assert isinstance(out["body"], str) and out["body"].strip()
    assert len(fake.prompts) == 1


def test_generate_inbound_prompts_never_leak_for_any_trigger():
    # exercise every trigger, including the grounded neighbour/supplier ones
    views = {
        "vet_followup": {"prior_advice": "rest and monitor"},
        "neighbour_concern": {"illness_severity": 3, "hunger": 90, "health": 10},
        "supplier_promo": {"catalogue": DEFAULT_CATALOGUE},
    }
    for key in TRIGGERS:
        fake = FakeModel()
        asyncio.run(generate_inbound(fake, key, view=views.get(key), seed=2))
        assert fake.prompts, f"{key}: model not called"
        low = fake.prompts[0].lower()
        for word in FORBIDDEN:
            assert word not in low, f"{word!r} leaked into trigger {key!r}"


def test_module_importable():
    assert personas is not None
