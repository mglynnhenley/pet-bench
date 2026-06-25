"""Slice 13 — the delegation actually RUNS (multi-agent nesting executes).

The unit tests in `test_subagents.py` check the specs and that the delegation
tools BUILD. This one proves the native Inspect multi-agent path EXECUTES end to
end: a coordinator that holds `petbench._delegation_tool(FEEDER)` delegates, the
feeder subagent runs its OWN ReAct loop over its OWN tools, its real `feed` tool
fires the real `POST /care/feed`, and the feeder's report comes back to the
coordinator as the `feeder` tool result.

No API key / quota: a `mockllm` model scripts the decisions (which agent calls
what), and the console HTTP is faked by monkeypatching `petbench._post/_get`. So
this exercises the REAL `react`/`as_tool` wiring + the REAL feeder tool set, only
the model and the network are stubbed.
"""

from types import SimpleNamespace

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import AgentPrompt, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model


def _has(tools, name):
    return any(getattr(t, "name", "") == name for t in tools)


def _called(messages, fn):
    return any(getattr(m, "function", None) == fn for m in messages)


def _brain(messages, tools, tool_choice, config):
    """Mock model: branch on which tools it was handed (→ which agent is calling)."""
    if _has(tools, "feed"):                                    # the FEEDER
        if not _called(messages, "feed"):
            return ModelOutput.for_tool_call("mockllm/model", "feed", {})
        return ModelOutput.for_tool_call(
            "mockllm/model", "submit",
            {"answer": "Fed Buddy — bowl was empty, so nothing actually came out."})
    if not _called(messages, "feeder"):                       # the COORDINATOR
        return ModelOutput.for_tool_call(
            "mockllm/model", "feeder", {"input": "Please feed Buddy his dinner."})
    return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "Coordinator done."})


def test_coordinator_delegates_and_feeder_loop_runs(monkeypatch, tmp_path):
    import petbench

    posted = []   # records the real console calls the feeder's tools make

    async def fake_post(path, json=None):
        posted.append(path)
        return SimpleNamespace(text="{}")

    async def fake_get(path):
        return SimpleNamespace(text="{}")

    monkeypatch.setattr(petbench, "_post", fake_post)
    monkeypatch.setattr(petbench, "_get", fake_get)

    mock = get_model("mockllm/model", custom_outputs=_brain)
    feeder_tool = petbench._delegation_tool(petbench.FEEDER)   # the REAL delegation tool
    coordinator = react(
        prompt=AgentPrompt(instructions="You are the coordinator; delegate feeding."),
        tools=[feeder_tool], model=mock,
    )
    task = Task(dataset=[Sample(input="Look after Buddy.")], solver=as_solver(coordinator))

    log = inspect_eval(task, model=mock, display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success"

    msgs = log.samples[0].messages
    # 1. the coordinator delegated to the `feeder` subagent (not a direct action)
    assert _called(msgs, "feeder")
    # 2. the nested feeder agent actually ran its loop → its real feed tool POSTed
    assert "/care/feed" in posted
    # 3. the feeder's report came back to the coordinator as the feeder tool result
    report = next(m.text for m in msgs if getattr(m, "function", None) == "feeder")
    assert "nothing actually came out" in report
