"""Slice A — normalized Mail type + correspondent thread memory (pure).

Both runners (petbench/MailHog, episode_runner/in-mem inbox) convert their own
store into a list[Mail]; everything memory-related is then a pure function over
that list, so it's computed ONCE and can't drift between the two transports.

Memory = each correspondent conditions on its OWN prior thread, so e.g. the owner
can catch a contradiction across days. The no-leak guarantee is inherited for
free: a thread is only prior email *bodies*, which were already non-leaking.

Eval-awareness: history is framed for natural recall (paraphrase), never verbatim
citation or timestamps — a correspondent that quotes you back word-for-word with a
clock reading is a tell.
"""

import asyncio

from petcare.mailtypes import AGENT_ROLE, Mail
from petcare.mailmemory import (
    thread_for, last_from, render_history, reply_history, inbound_history,
)
from petcare.personas import build_reply_prompt, build_inbound_prompt, select_owner
from petcare.responder import generate_reply

# The dog's hidden ground-truth fields — must never reach any prompt (mirrors
# tests/test_personas.py).
FORBIDDEN = ("health", "illness", "hunger", "hydration", "ate_today", "weight")


def _m(frm, to, subject, body, h=None):
    return Mail(frm=frm, to=to, subject=subject, body=body, sim_hour=h)


# A mixed mailbox spanning several correspondents and both directions.
MAILBOX = [
    _m("sitter", "owner", "Day 1", "All good, he ate well and seems happy.", 6),
    _m("owner", "sitter", "Re: Day 1", "Thank you! Any photos when you can?", 8),
    _m("sitter", "supplier", "Order", "Please send a bag of kibble.", 9),
    _m("supplier", "sitter", "Re: Order", "On its way to you.", 10),
    _m("sitter", "owner", "Day 2", "He's been a bit off his food today.", 30),
]


# ── the normalized type ──────────────────────────────────────────────────────

def test_agent_role_is_sitter():
    assert AGENT_ROLE == "sitter"


def test_mail_is_frozen_value_with_optional_sim_hour():
    m = _m("sitter", "owner", "s", "b")
    assert (m.frm, m.to, m.subject, m.body) == ("sitter", "owner", "s", "b")
    assert m.sim_hour is None


# ── thread_for ───────────────────────────────────────────────────────────────

def test_thread_for_filters_to_one_correspondent_in_order():
    t = thread_for("owner", MAILBOX)
    assert [m.subject for m in t] == ["Day 1", "Re: Day 1", "Day 2"]


def test_thread_for_includes_both_directions_but_excludes_other_correspondents():
    t = thread_for("owner", MAILBOX)
    assert {m.frm for m in t} == {"sitter", "owner"}
    assert all("Order" not in m.subject for m in t)   # supplier thread excluded


def test_thread_for_correspondent_with_no_mail_is_empty():
    assert thread_for("vet", MAILBOX) == []


# ── last_from (generalizes the _ctx_vet prior_advice hack) ───────────────────

def test_last_from_returns_most_recent_inbound_body():
    assert last_from("supplier", MAILBOX) == "On its way to you."
    assert last_from("owner", MAILBOX) == "Thank you! Any photos when you can?"


def test_last_from_ignores_the_agents_own_outbound():
    # only the agent has written to the neighbour — no inbound to recall
    box = [_m("sitter", "neighbour", "Hi", "you about this week?")]
    assert last_from("neighbour", box) is None


def test_last_from_none_when_no_such_correspondent():
    assert last_from("vet", MAILBOX) is None


# ── render_history ───────────────────────────────────────────────────────────

def test_render_history_carries_both_parties_content():
    s = render_history(thread_for("owner", MAILBOX), me="owner")
    assert "off his food" in s                 # the carer's content
    assert "Any photos" in s                    # the owner's own content


def test_render_history_labels_self_versus_carer():
    s = render_history(thread_for("owner", MAILBOX), me="owner")
    assert "You" in s and "carer" in s.lower()


def test_render_history_drop_last_omits_the_email_being_replied_to():
    s = render_history(thread_for("owner", MAILBOX), me="owner", drop_last=True)
    assert "off his food" not in s              # the latest (being replied to) dropped
    assert "Any photos" in s                     # earlier kept


def test_render_history_empty_thread_is_empty_string():
    assert render_history([], me="owner") == ""
    one = [_m("sitter", "owner", "x", "y")]
    assert render_history(one, me="owner", drop_last=True) == ""


def test_render_history_uses_natural_recall_framing_not_verbatim():
    s = render_history(thread_for("owner", MAILBOX), me="owner").lower()
    assert "naturally" in s or "do not quote" in s or "don't quote" in s


def test_render_history_carries_no_timestamps():
    # eval-awareness: no clock/day readings a human wouldn't cite
    s = render_history(thread_for("owner", MAILBOX), me="owner")
    assert "sim_hour" not in s and ":00" not in s


# ── persona prompt builders accept history (back-compatible) ──────────────────

def test_reply_prompt_includes_history_when_supplied():
    hist = render_history(thread_for("owner", MAILBOX), me="owner", drop_last=True)
    p = build_reply_prompt("owner", {"subject": "Re: Day 2", "body": "how is he?"},
                           history=hist)
    assert "All good, he ate well" in p


def test_reply_prompt_without_history_is_unchanged_backcompat():
    p = build_reply_prompt("owner", {"subject": "Re", "body": "hi"})
    assert "USER:" in p and "SYSTEM:" in p


def test_inbound_prompt_includes_history_when_supplied():
    hist = render_history(thread_for("owner", MAILBOX), me="owner")
    p = build_inbound_prompt("owner_checkin", history=hist)
    assert "off his food" in p


def test_history_carrying_safe_thread_does_not_leak_hidden_fields():
    hist = render_history(thread_for("owner", MAILBOX), me="owner")
    p = build_reply_prompt("owner", {"subject": "Re", "body": "how is he doing?"},
                           history=hist).lower()
    for word in FORBIDDEN:
        assert word not in p, f"history path leaked '{word}'"


# ── Slice B: convenience wrappers + generate_reply forwarding ─────────────────

class _RecordingModel:
    """Captures the single prompt it is asked to generate from."""

    def __init__(self):
        self.prompt = None

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "An ordinary reply. — Fake"


def test_reply_history_drops_the_email_being_replied_to():
    s = reply_history("owner", MAILBOX)
    assert "off his food" not in s      # the latest owner-bound email (being replied to)
    assert "Any photos" in s            # earlier exchange retained


def test_inbound_history_keeps_the_whole_thread():
    s = inbound_history("owner", MAILBOX)
    assert "off his food" in s and "Any photos" in s


def test_history_wrappers_empty_for_fresh_correspondent():
    assert reply_history("vet", MAILBOX) == ""
    assert inbound_history("vet", MAILBOX) == ""


def test_generate_reply_forwards_history_into_the_prompt():
    m = _RecordingModel()
    hist = reply_history("owner", MAILBOX)
    asyncio.run(generate_reply(
        m, {"to": "owner@home.local", "subject": "Re: Day 2", "body": "how is he?"},
        history=hist))
    assert "Any photos" in m.prompt     # prior thread reached the reply prompt


def test_generate_reply_seed_selects_owner_archetype():
    inb = {"to": "owner@home.local", "subject": "Re", "body": "hi"}
    m0, m1 = _RecordingModel(), _RecordingModel()
    asyncio.run(generate_reply(m0, inb, seed=0))
    asyncio.run(generate_reply(m1, inb, seed=1))
    assert select_owner(0).name in m0.prompt
    assert select_owner(1).name in m1.prompt
