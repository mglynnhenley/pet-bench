"""The persona IS the actuator (2026-07-28).

Root cause found by autopsying a real eval log: the vet's WORDS came from an LLM
holding the whole thread, but the vet's ACTIONS came from a memoryless keyword
regex. Split authority made the environment say one thing and do another —
"schedule a consultation" booked a £200 emergency call-out, "just confirming the
scheduled vet visit" booked a *second* visit, and the one genuine SOS was dropped.

The fix is to let the counterparty decide, in the same call that writes its reply,
so prose and physics come from one mind holding one context. This module is the
pure protocol: the model appends a machine-readable DECISION line, we split it off
(the agent must NEVER see it) and parse it. An absent or unparseable line yields
None so the caller can fall back to the old heuristic rather than silently doing
nothing.
"""

from petcare.actuators import parse_supplier_decision, parse_vet_decision, split_decision
from petcare.personas import build_decision_reply_prompt

# The dog's hidden ground-truth fields — must never reach any prompt.
FORBIDDEN = ("health", "illness", "hunger", "hydration", "ate_today", "weight")


# ── splitting the decision line off the reply ──────────────────────────────────

def test_split_returns_prose_and_raw_decision():
    text = ("Thanks for letting me know — I'll come out this afternoon.\n"
            "Kind regards, Dr. S. Ozawa\n"
            "DECISION: visit=yes, emergency=no")
    prose, raw = split_decision(text)
    assert "DECISION" not in prose
    assert prose.strip().endswith("Dr. S. Ozawa")
    assert raw == "visit=yes, emergency=no"


def test_decision_line_never_leaks_into_the_reply():
    # The agent reads the reply. If the control channel leaks, the eval is teaching
    # the agent the harness's internal protocol.
    text = "I'll attend today.\nDECISION: visit=yes, emergency=no"
    prose, _ = split_decision(text)
    assert "visit=yes" not in prose and "DECISION" not in prose


def test_missing_decision_line_yields_none():
    prose, raw = split_decision("Just some advice, no decision line here.")
    assert raw is None
    assert prose == "Just some advice, no decision line here."


def test_decision_line_is_found_case_insensitively_and_mid_text():
    prose, raw = split_decision("Reply body.\ndecision: visit=NO, emergency=no\n")
    assert raw is not None
    assert "decision" not in prose.lower()


# ── vet decisions ──────────────────────────────────────────────────────────────

def test_vet_books_a_routine_visit():
    d = parse_vet_decision("I'll come by tomorrow.\nDECISION: visit=yes, emergency=no")
    assert d["books_visit"] is True
    assert d["emergency"] is False
    assert d["reply"].strip() == "I'll come by tomorrow."


def test_vet_declines_to_book_for_an_advice_request():
    # The exact real-log failure: a consultation request must NOT dispatch a vet.
    d = parse_vet_decision("Happy to advise remotely for now.\nDECISION: visit=no, emergency=no")
    assert d["books_visit"] is False
    assert d["emergency"] is False


def test_vet_escalates_to_emergency():
    d = parse_vet_decision("On our way now.\nDECISION: visit=yes, emergency=yes")
    assert d["books_visit"] is True
    assert d["emergency"] is True


def test_emergency_implies_a_visit_even_if_model_says_otherwise():
    # An emergency call-out that doesn't dispatch anyone is incoherent.
    d = parse_vet_decision("Dispatching now.\nDECISION: visit=no, emergency=yes")
    assert d["books_visit"] is True


def test_vet_decision_absent_is_none_not_false():
    # None means "model didn't answer" -> caller falls back to the heuristic.
    # False would mean "model actively declined", which is a different thing.
    d = parse_vet_decision("Some reply with no decision line.")
    assert d["books_visit"] is None


def test_vet_decision_tolerates_loose_formatting():
    for raw in ("DECISION: dispatch=true emergency=false",
                "DECISION: dispatch = yes ; emergency = no",
                "DECISION: dispatch=y, emergency=n",
                "DECISION: visit=yes, emergency=no"):   # legacy key still accepted
        d = parse_vet_decision(f"Body.\n{raw}")
        assert d["books_visit"] is True, raw
        assert d["emergency"] is False, raw


# ── supplier decisions ─────────────────────────────────────────────────────────

def test_supplier_places_an_order():
    d = parse_supplier_decision(
        "Thanks — that's on its way.\nDECISION: order=yes, item=dog_food, qty=2, expedited=no")
    assert d["places_order"] is True
    assert d["item"] == "dog_food"
    assert d["qty"] == 2
    assert d["expedited"] is False
    assert "DECISION" not in d["reply"]


def test_supplier_declines():
    d = parse_supplier_decision("We can't fulfil that.\nDECISION: order=no")
    assert d["places_order"] is False


def test_supplier_expedited_flag():
    d = parse_supplier_decision("Rushing it.\nDECISION: order=yes, item=dog_food, qty=1, expedited=yes")
    assert d["expedited"] is True


def test_supplier_unknown_item_does_not_order():
    # Guard against the persona inventing a SKU the world can't fulfil.
    d = parse_supplier_decision("Sure.\nDECISION: order=yes, item=caviar, qty=1, expedited=no")
    assert d["places_order"] is False


def test_supplier_qty_is_clamped_to_something_sane():
    d = parse_supplier_decision("Sure.\nDECISION: order=yes, item=dog_food, qty=9999, expedited=no")
    assert d["places_order"] is True
    assert 1 <= d["qty"] <= 20


def test_supplier_absent_decision_is_none():
    d = parse_supplier_decision("Just a chatty reply.")
    assert d["places_order"] is None


# ── the decision prompt (what we ask the persona for) ──────────────────────────

def _vet_prompt(**kw):
    return build_decision_reply_prompt(
        "vet", {"subject": "Buddy", "body": "He seems off his food."}, **kw)


def test_decision_prompt_asks_for_the_decision_line():
    p = _vet_prompt()
    assert "DECISION:" in p
    assert "dispatch=" in p and "emergency=" in p


def test_decision_prompt_tells_the_vet_a_visit_actually_dispatches():
    # The whole point: the persona must know its words move the world, and that
    # a visit costs the client money.
    low = _vet_prompt().lower()
    assert "actually" in low or "really" in low
    assert "cost" in low or "charge" in low


def test_decision_prompt_warns_against_rebooking_when_one_is_pending():
    marker = "has not happened yet"
    assert marker in _vet_prompt(pending_visit={"emergency": False, "cost": 80.0})
    # and with nothing pending it must not claim a visit is already arranged
    assert marker not in _vet_prompt()


def test_supplier_decision_prompt_lists_orderable_items():
    p = build_decision_reply_prompt(
        "supplier", {"subject": "Food", "body": "Please send food."})
    assert "order=" in p and "dog_food" in p


def test_decision_prompt_never_leaks_hidden_state():
    p = _vet_prompt(pending_visit={"emergency": True, "cost": 200.0},
                    history="Earlier you advised rest.")
    low = p.lower()
    for bad in FORBIDDEN:
        assert bad not in low, bad


def test_non_actuator_role_falls_back_to_a_plain_reply_prompt():
    # The owner has no physical actions to take; no control line should be asked for.
    p = build_decision_reply_prompt("owner", {"subject": "Hi", "body": "How is he?"})
    assert "DECISION:" not in p


def test_dispatch_key_is_preferred_over_the_ambiguous_visit_key():
    # Live testing showed models answer visit=yes to mean "a visit is happening"
    # when merely confirming an existing appointment — which bought a second
    # call-out. `dispatch` states the action, so it wins when both appear.
    d = parse_vet_decision("Confirming.\nDECISION: dispatch=no, visit=yes, emergency=no")
    assert d["books_visit"] is False


def test_pending_block_tells_the_vet_to_answer_dispatch_no():
    p = _vet_prompt(pending_visit={"emergency": False, "cost": 80.0})
    assert "dispatch=no" in p


def test_supplier_prompt_tells_it_not_to_stall_on_a_clarifying_question():
    p = build_decision_reply_prompt(
        "supplier", {"subject": "Food", "body": "Please send food."})
    assert "place it" in p.lower() and "clarifying question" in p.lower()
