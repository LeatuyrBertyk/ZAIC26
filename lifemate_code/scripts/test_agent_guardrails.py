"""Unit tests for LifeMateAgent v5 guardrails (CPU, no model needed):

    python -X utf8 scripts/test_agent_guardrails.py

1. Sanitizer: unknown keys dropped, "26" -> 26, "500.000" -> 500000, "7:30" -> "07:30", naive date-time -> +07:00 with seconds
2. sys.ask_missing_info: READ target allowed when params are required (flights.search/date); contacts.search blocked
3. Schema-invalid calls are blocked BEFORE env.call (never recorded)
4. Hallucinated x-ref ID blocked; ID returned by another app's result accepted
5. Identical WRITE re-issued is blocked
6. Out-of-scope shortcut: first proposal targets a real but uninstalled app -> [] immediately
7. Context budget: oldest history dropped instead of crashing / cutting the user's request
8. Time budget: deadline in the past -> no LLM call at all
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DATA_ROOT / "train"))

from lifemate.agent import BaseLLMBackend, LifeMateAgent, StepEnsembleLLMBackend  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402


class Scripted(BaseLLMBackend):
    """Returns one pre-written list of calls per step."""

    def __init__(self, steps: list[list[dict]], token_counter=None):
        self.steps = steps
        self.n = 0
        self.seen_messages: list[list[dict]] = []
        if token_counter:
            self.count_tokens = token_counter

    def generate(self, messages, tools):
        self.seen_messages.append(messages)
        out = self.steps[self.n] if self.n < len(self.steps) else []
        self.n += 1
        return "", out


def load(qid: str) -> dict:
    for line in open(DATA_ROOT / "train" / "train.jsonl", encoding="utf-8"):
        if line.strip() and json.loads(line)["id"] == qid:
            return json.loads(line)
    raise KeyError(qid)


def tmap(s: dict) -> dict:
    m = {h["audio"]: {"primary": h.get("audio_text") or ""} for h in s.get("history") or [] if h.get("audio")}
    m[s["audio"]] = {"primary": s.get("audio_text") or ""}
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    agent = LifeMateAgent(llm=Scripted([]), catalog=catalog)

    # 1. sanitizer
    a = agent._sanitize_arguments("smart_home.climate_set", {"device_id": "ac_1", "mode": "COOL", "temperature": "26", "foo": 1})
    assert a == {"device_id": "ac_1", "mode": "cool", "temperature": 26}, a
    a = agent._sanitize_arguments("banking.transfer", {"from_account_id": "a", "beneficiary_id": "b", "amount": "500.000", "note": ""})
    assert a == {"from_account_id": "a", "beneficiary_id": "b", "amount": 500000}, a
    a = agent._sanitize_arguments("alarm.create", {"time": "7:30", "repeat": "once"})
    assert a["time"] == "07:30", a
    a = agent._sanitize_arguments("calendar.update_event", {"event_id": "e1", "start": "2026-11-29T09:00", "end": "2026-11-29 10:00:00"})
    assert a["start"] == "2026-11-29T09:00:00+07:00" and a["end"] == "2026-11-29T10:00:00+07:00", a
    assert agent._sanitize_arguments("alarm.list", {"x": 1}) == {}
    print("PASS 1 sanitizer")

    # 2. ask_missing_info target rules
    allowed = {"flights.search", "contacts.search", "sms.send"}
    ok = agent._sanitize_ask({"tool_name": "flights.search", "params": ["date", "date", "bogus"]})
    assert ok["params"] == ["date"] and agent._ask_target_error(ok, allowed) is None, ok
    bad = agent._sanitize_ask({"tool_name": "contacts.search", "params": ["query"]})
    assert agent._ask_target_error(bad, allowed) is not None
    w = agent._sanitize_ask({"tool_name": "sms.send", "params": ["contact_id"]})
    assert agent._ask_target_error(w, allowed) is None
    print("PASS 2 ask_missing_info targets")

    # 3 + 4 + 5 on a real sample session
    s = load("train_0001")
    llm = Scripted(
        [
            [{"name": "weather.get_forecast", "arguments": {}}],  # schema error -> blocked, not recorded
            [{"name": "calendar.update_event", "arguments": {"event_id": "evt_guessed", "start": "2026-11-29T09:00"}}],  # x-ref
            [{"name": "smart_home.get_status", "arguments": {}}],
            [{"name": "alarm.create", "arguments": {"time": "6:00", "repeat": "once"}}],
            [{"name": "alarm.create", "arguments": {"time": "06:00", "repeat": "once"}}],  # identical write -> blocked
        ]
    )
    env = ToolEnv("train", "train_0001", root=DATA_ROOT)
    LifeMateAgent(llm=llm, catalog=catalog).run_sample(env, s, tmap(s))
    names = [c["name"] for c in env.get_tool_calls()]
    assert names == ["smart_home.get_status", "alarm.create"], env.get_tool_calls()
    print("PASS 3-5 schema / x-ref / duplicate-write guards ->", names)

    # 4b. ID returned by another app's tool result (history) is accepted (gold train_1005 uses c_778 from sms.read_recent)
    s = load("train_1005")
    gold = s["label"]["tool_calls"]
    env = ToolEnv("train", "train_1005", root=DATA_ROOT)
    LifeMateAgent(llm=Scripted([[c] for c in gold]), catalog=catalog).run_sample(env, s, tmap(s))
    assert [c["name"] for c in env.get_tool_calls()] == [c["name"] for c in gold], env.get_tool_calls()
    print("PASS 4b cross-app x-ref accepted")

    # 6. out-of-scope shortcut
    s = load("train_0001")
    env = ToolEnv("train", "train_0001", root=DATA_ROOT)
    llm = Scripted([[{"name": "ride_hailing.book", "arguments": {"pickup": "a", "dropoff": "b"}}], [{"name": "alarm.list", "arguments": {}}]])
    LifeMateAgent(llm=llm, catalog=catalog).run_sample(env, s, tmap(s))
    assert env.get_tool_calls() == [] and llm.n == 1, (env.get_tool_calls(), llm.n)
    print("PASS 6 out-of-scope -> []")

    # 7. context budget: fake counter = number of characters / 4
    s = next(
        json.loads(l)
        for l in open(DATA_ROOT / "train" / "train.jsonl", encoding="utf-8")
        if l.strip() and len(json.loads(l).get("history") or []) >= 6
    )
    counter = lambda msgs, tools: len(json.dumps(msgs, ensure_ascii=False)) // 4  # noqa: E731
    llm = Scripted([], token_counter=counter)
    ag = LifeMateAgent(llm=llm, catalog=catalog, max_new_tokens=384, max_context_tokens=1800)
    env = ToolEnv("train", s["id"], root=DATA_ROOT)
    ag.run_sample(env, s, tmap(s))
    if llm.seen_messages:
        seen = llm.seen_messages[0]
        assert seen[-1]["role"] == "user" and seen[-1]["content"], "final user request must survive trimming"
        assert counter(seen, []) <= ag.prompt_budget
        print(f"PASS 7 context budget (history trimmed to {len(seen)} messages)")
    else:
        print("PASS 7 context budget (prompt could not fit -> stopped cleanly without crashing)")

    # 8. deadline
    llm = Scripted([[{"name": "alarm.list", "arguments": {}}]])
    env = ToolEnv("train", "train_0001", root=DATA_ROOT)
    LifeMateAgent(llm=llm, catalog=catalog).run_sample(env, load("train_0001"), {}, deadline=time.perf_counter() - 1)
    assert llm.n == 0 and env.get_tool_calls() == []
    print("PASS 8 time budget")
    # 9. invented amount after a vague quantity ("chút tiền") -> sys.ask_missing_info(amount); real val failure train_0268
    s = load("train_0268")
    gold = s["label"]["tool_calls"]
    proposed = [dict(c) for c in gold[:2]] + [{"name": "ewallet.transfer", "arguments": {"contact_id": gold[0]["arguments"].get("contact_id", "c_04"), "amount": 500000}}]
    # use the contact id the real search returns
    env = ToolEnv("train", "train_0268", root=DATA_ROOT)
    first = env.call("contacts.search", {"query": "Tuấn"})
    cid = first["contacts"][0]["id"]
    env = ToolEnv("train", "train_0268", root=DATA_ROOT)
    llm = Scripted([[{"name": "contacts.search", "arguments": {"query": "Tuấn"}}],
                    [{"name": "alarm.create", "arguments": gold[1]["arguments"]}],
                    [{"name": "ewallet.transfer", "arguments": {"contact_id": cid, "amount": 500000}}]])
    LifeMateAgent(llm=llm, catalog=catalog).run_sample(env, s, tmap(s))
    got = env.get_tool_calls()
    assert [c["name"] for c in got] == [c["name"] for c in gold], got
    assert got[-1]["arguments"] == gold[-1]["arguments"], got[-1]
    print("PASS 9 vague amount -> ask_missing_info")
    # ...and a clearly stated amount is left alone
    assert not LifeMateAgent.amount_is_missing("ewallet.transfer", {"amount": 50000}, "chuyển ví cho Tuấn năm mươi nghìn")
    assert LifeMateAgent.amount_is_missing("ewallet.transfer", {"amount": 50000}, "gửi Tuấn ít tiền qua ví")
    print("PASS 9 vague amount -> ask_missing_info")

    # 10. Empty read retry cap (Trick #1.3: stop after 2 empty searches on the same tool)
    s = load("train_0041")
    env = ToolEnv("train", "train_0041", root=DATA_ROOT)
    llm = Scripted(
        [
            [{"name": "contacts.search", "arguments": {"query": "KhôngTồnTại1"}}],
            [{"name": "contacts.search", "arguments": {"query": "KhôngTồnTại2"}}],
            [{"name": "contacts.search", "arguments": {"query": "KhôngTồnTại3"}}],  # 3rd empty search -> blocked!
        ]
    )
    LifeMateAgent(llm=llm, catalog=catalog, max_empty_read_retries=2).run_sample(env, s, tmap(s))
    assert len(env.get_tool_calls()) == 2, env.get_tool_calls()
    print("PASS 10 empty read retry cap (3rd empty search blocked before ToolEnv)")

    # 11. Step-level majority voting ensemble with early-exit consensus (Trick #3.1)
    b1 = Scripted([[{"name": "alarm.create", "arguments": {"time": "06:00", "repeat": "once"}}]])
    b2 = Scripted([[{"name": "alarm.create", "arguments": {"time": "06:00", "repeat": "once"}}]])
    b3 = Scripted([[{"name": "alarm.create", "arguments": {"time": "07:00", "repeat": "once"}}]])
    ens = StepEnsembleLLMBackend([b1, b2, b3], early_exit_consensus=True)
    _, voted_calls = ens.generate([], [])
    assert voted_calls == [{"name": "alarm.create", "arguments": {"time": "06:00", "repeat": "once"}}]
    assert b3.n == 0, "Early-exit consensus should skip backend 3 when backend 1 and 2 agree!"
    print("PASS 11 step-level majority voting ensemble + early-exit consensus")

    print("ALL GUARDRAIL TESTS PASSED")


if __name__ == "__main__":
    main()
