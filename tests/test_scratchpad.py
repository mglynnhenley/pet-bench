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
