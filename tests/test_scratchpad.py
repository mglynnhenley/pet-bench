"""Slice: the private 'unmonitored' scratchpad tool (petbench.scratchpad).

The notepad is one accumulating string in the Inspect store under the exact key
"scratchpad". These tests exercise:
  - append-then-read returns the appended text,
  - multiple appends accumulate across simulated turns,
  - reading an empty notepad is handled gracefully,
both at the pure-helper level and against a real Inspect store context.
"""

import asyncio

import pytest

from inspect_ai.util import Store
from inspect_ai.util._store import init_subtask_store, store

from petbench import _SCRATCHPAD_KEY, _scratchpad_op, scratchpad


# ── Pure helper: _scratchpad_op(notepad, mode, text) -> (new_notepad, response) ──

def test_read_empty_is_graceful():
    notepad, response = _scratchpad_op("", "read")
    assert notepad == ""
    assert "empty" in response.lower()


def test_append_then_read():
    notepad, resp1 = _scratchpad_op("", "append", "buddy refused breakfast")
    assert "buddy refused breakfast" in notepad
    assert resp1  # a confirmation string
    read_pad, resp2 = _scratchpad_op(notepad, "read")
    assert "buddy refused breakfast" in resp2
    assert read_pad == notepad


def test_multiple_appends_accumulate():
    pad = ""
    for note in ["note one", "note two", "note three"]:
        pad, _ = _scratchpad_op(pad, "append", note)
    assert "note one" in pad
    assert "note two" in pad
    assert "note three" in pad
    # a separator keeps notes distinct
    assert pad.count("note") == 3
    _, readback = _scratchpad_op(pad, "read")
    assert readback == pad


def test_unknown_mode_is_noop():
    pad = "existing"
    new, resp = _scratchpad_op(pad, "delete", "x")
    assert new == pad  # unchanged → tool won't rewrite the store
    assert "unknown" in resp.lower()
    assert "overwrite" in resp.lower()  # the valid modes (incl. overwrite) are advertised


# ── The size cap + overwrite mode ───────────────────────────────────────────────
# The pad is capped (SCRATCHPAD_MAX_CHARS). An append that would overflow is
# rejected (pad unchanged) and the agent is pointed at `overwrite`, which replaces
# the whole pad — its escape valve, and how it condenses/rewrites stale thinking.

def test_default_limit_is_generous():
    from petcare.scratchpad import SCRATCHPAD_MAX_CHARS
    assert SCRATCHPAD_MAX_CHARS >= 1000  # generous default for normal use
    new, _ = _scratchpad_op("", "append", "a normal check-in note")
    assert "a normal check-in note" in new  # a normal note fits under the default


def test_append_within_limit_reports_usage():
    new, resp = _scratchpad_op("", "append", "z" * 30, max_chars=50)
    assert new == "z" * 30
    assert "30/50" in resp  # the agent is told how full the pad is


def test_append_over_limit_is_rejected():
    pad = "x" * 40
    new, resp = _scratchpad_op(pad, "append", "y" * 20, max_chars=50)
    assert new == pad  # nothing written — pad unchanged
    assert "full" in resp.lower()
    assert "overwrite" in resp.lower()  # told how to make room


def test_overwrite_replaces_whole_pad():
    pad, _ = _scratchpad_op("", "append", "first thought", max_chars=200)
    pad, _ = _scratchpad_op(pad, "append", "second thought", max_chars=200)
    new, resp = _scratchpad_op(pad, "overwrite", "condensed: 2 thoughts", max_chars=200)
    assert new == "condensed: 2 thoughts"
    assert "first thought" not in new  # previous thinking is gone
    assert "overwritten" in resp.lower()


def test_overwrite_frees_room_after_full():
    pad = "x" * 48
    rejected, _ = _scratchpad_op(pad, "append", "more", max_chars=50)
    assert rejected == pad  # full
    condensed, _ = _scratchpad_op(pad, "overwrite", "short", max_chars=50)
    assert condensed == "short"
    grown, _ = _scratchpad_op(condensed, "append", "ok", max_chars=50)
    assert "ok" in grown  # room again


def test_overwrite_over_limit_is_rejected():
    pad = "keep me"
    new, resp = _scratchpad_op(pad, "overwrite", "y" * 100, max_chars=50)
    assert new == pad  # unchanged on reject
    assert "too long" in resp.lower()


def test_overwrite_empty_clears():
    pad, _ = _scratchpad_op("", "append", "something", max_chars=200)
    new, resp = _scratchpad_op(pad, "overwrite", "", max_chars=200)
    assert new == ""
    assert "clear" in resp.lower()


# ── Against a real Inspect store context (the way the tool actually runs) ────────

def _fresh_store() -> Store:
    s = Store()
    init_subtask_store(s)
    return s


def test_tool_append_then_read_via_store():
    async def run():
        _fresh_store()
        execute = scratchpad()
        await execute(mode="append", text="dispensed food, watching appetite")
        out = await execute(mode="read", text="")
        assert "dispensed food, watching appetite" in out
        # persisted under the exact key the eval log captures
        assert store().get(_SCRATCHPAD_KEY, "") != ""
    asyncio.run(run())


def test_tool_appends_accumulate_across_turns_via_store():
    async def run():
        s = _fresh_store()
        execute = scratchpad()
        # simulate separate check-ins re-binding the same store
        await execute(mode="append", text="day 1 plan")
        init_subtask_store(s)  # next turn, same backing store
        await execute(mode="append", text="day 2 plan")
        out = await execute(mode="read", text="")
        assert "day 1 plan" in out
        assert "day 2 plan" in out
    asyncio.run(run())


def test_tool_overwrite_replaces_via_store():
    async def run():
        _fresh_store()
        execute = scratchpad()
        await execute(mode="append", text="early rough notes — buddy maybe off his food")
        await execute(mode="overwrite", text="condensed: buddy fine now")
        out = await execute(mode="read", text="")
        assert "condensed: buddy fine now" in out
        assert "early rough notes" not in out  # overwrite cleared the old pad
        assert store().get(_SCRATCHPAD_KEY, "") == "condensed: buddy fine now"
    asyncio.run(run())


def test_tool_read_empty_via_store():
    async def run():
        _fresh_store()
        execute = scratchpad()
        out = await execute(mode="read", text="")
        assert "empty" in out.lower()
        # reading must not create content
        assert store().get(_SCRATCHPAD_KEY, "") == ""
    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
