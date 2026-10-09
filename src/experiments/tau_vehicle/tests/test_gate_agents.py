import pytest

from experiments.tau_vehicle import gate_agents as ga
from experiments.tau_vehicle.gate_agents import VehicleContextGateAgent, VehicleContextTailAgent, VehicleGateAgent
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.vehicle.environment import get_environment

LLM = "gpt-5.4-mini"


@pytest.fixture
def env():
    e = get_environment()
    e.run_env_function_calls([EnvFunctionCall(env_type="assistant", func_name="set_vehicle_state",
                                              arguments={"vehicle": {"gear_selection": "D", "perf_vehicle_speed_kmh": 60.0}})])
    return e


def unlock_and_temp():
    return AssistantMessage(role="assistant", content=None, cost=0.0, tool_calls=[
        ToolCall(id="a", name="set_door_lock", arguments={"locked": False}),
        ToolCall(id="b", name="set_climate", arguments={"hvac_temperature_set_c": 20}),
    ])


def only_temp():
    return AssistantMessage(role="assistant", content=None, cost=0.0,
                            tool_calls=[ToolCall(id="b", name="set_climate", arguments={"hvac_temperature_set_c": 20})])


@pytest.mark.parametrize("cls", [VehicleGateAgent, VehicleContextGateAgent])
def test_gate_revise_then_allow(env, cls, monkeypatch):
    agent = cls(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    proposals = [unlock_and_temp(), only_temp()]
    seen = []

    def fake_generate(model, tools, messages, call_name, **kw):
        seen.append([m.content for m in messages if m.role == "system"])
        return proposals.pop(0)

    monkeypatch.setattr(ga, "generate", fake_generate)
    import experiments.tau_vehicle.context_agent as ca
    monkeypatch.setattr(ca, "generate", fake_generate)
    from tau2.agent import llm_agent as la
    monkeypatch.setattr(la, "generate", fake_generate)
    state = agent.get_init_state()
    msg, state = agent.generate_next_message(UserMessage(role="user", content="unlock and set 20"), state)
    assert [tc.name for tc in msg.tool_calls] == ["set_climate"]
    assert msg.raw_data["safety_gate"][0]["action"] == "REVISE"
    assert msg.raw_data["safety_gate"][0]["blocked"][0]["rules"] == ["S2"]
    assert any("SAFETY GATE" in c for c in seen[1])  # second call saw the gate note


@pytest.mark.parametrize("cls", [VehicleGateAgent, VehicleContextGateAgent])
def test_gate_blocks_when_model_insists(env, cls, monkeypatch):
    agent = cls(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    fake = lambda model, tools, messages, call_name, **kw: unlock_and_temp()
    monkeypatch.setattr(ga, "generate", fake)
    import experiments.tau_vehicle.context_agent as ca
    monkeypatch.setattr(ca, "generate", fake)
    from tau2.agent import llm_agent as la
    monkeypatch.setattr(la, "generate", fake)
    state = agent.get_init_state()
    msg, _ = agent.generate_next_message(UserMessage(role="user", content="unlock and set 20"), state)
    assert [tc.name for tc in msg.tool_calls] == ["set_climate"]
    assert msg.raw_data["safety_gate"][-1]["action"] == "BLOCK"


def test_gate_sequential_child_lock_then_window(env):
    env.run_env_function_calls([EnvFunctionCall(env_type="assistant", func_name="set_vehicle_state",
                                                arguments={"vehicle": {"seat_occupancy": {"rear_child_seat_occupied": True}}})])
    calls = [ToolCall(id="1", name="set_child_lock", arguments={"enabled": True}),
             ToolCall(id="2", name="set_window", arguments={"position": "rear_left", "open_pct": 30})]
    allowed, blocked = ga.gate_check(env.tools.db, calls)
    assert len(allowed) == 2 and not blocked
    allowed, blocked = ga.gate_check(env.tools.db, calls[1:])  # window before child lock -> blocked
    assert blocked and blocked[0][1] == ["S3"]


def test_tail_agent_keeps_system_prompt_and_appends_block(env, monkeypatch):
    a = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    t = VehicleContextTailAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    assert t.system_prompt == a.system_prompt
    captured = {}

    def fake_generate(model, tools, messages, call_name, **kw):
        captured["messages"] = messages
        return AssistantMessage(role="assistant", content="ok", cost=0.0)

    monkeypatch.setattr(ga, "generate", fake_generate)
    state = t.get_init_state()
    t.generate_next_message(UserMessage(role="user", content="hi"), state)
    msgs = captured["messages"]
    assert msgs[0].content == a.system_prompt
    assert msgs[-1].role == "system" and "<vehicle_state>" in msgs[-1].content and '"gear_selection": "D"' in msgs[-1].content
    assert all("<vehicle_state>" not in (m.content or "") for m in state.messages)  # block is not stored in history


def test_lastmsg_agent_patches_only_last_message(env, monkeypatch):
    from experiments.tau_vehicle.gate_agents import VehicleContextLastMsgAgent
    from tau2.data_model.message import ToolMessage
    a = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    t = VehicleContextLastMsgAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=LLM, llm_args={})
    captured = []

    def fake_generate(model, tools, messages, call_name, **kw):
        captured.append(messages)
        return AssistantMessage(role="assistant", content="ok", cost=0.0)

    monkeypatch.setattr(ga, "generate", fake_generate)
    state = t.get_init_state()
    t.generate_next_message(UserMessage(role="user", content="hi"), state)
    t.generate_next_message(ToolMessage(id="1", role="tool", content='{"x": 1}', requestor="assistant"), state)
    first, second = captured
    assert first[0].content == a.system_prompt
    assert first[-1].role == "user" and first[-1].content.startswith("hi\n\n<vehicle_state>")
    assert second[-1].role == "tool" and second[-1].content.startswith('{"x": 1}\n\n<vehicle_state>')
    assert second[-3].content == "hi"  # earlier user message kept its original content (second[-2] is the assistant reply)
    assert all("<vehicle_state>" not in (m.content or "") for m in state.messages)
