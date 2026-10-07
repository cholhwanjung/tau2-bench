import pytest

from experiments.tau_vehicle import context_agent as ca
from experiments.tau_vehicle.context_agent import VehicleContextAgent, STATE_BLOCK
from experiments.tau_vehicle.noop_agent import NoOpAgent
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.vehicle.environment import get_environment

LLM = "gpt-5.4-mini"


@pytest.fixture
def env():
    return get_environment()


@pytest.fixture
def agents(env):
    tools = env.get_tools()
    policy = env.get_policy()
    a = LLMAgent(tools=tools, domain_policy=policy, llm=LLM, llm_args={"temperature": 0})
    b = VehicleContextAgent(tools=tools, domain_policy=policy, llm=LLM, llm_args={"temperature": 0})
    return a, b


def state_tool_output(env) -> str:
    return env.get_response(ToolCall(id="x", name="get_vehicle_state", arguments={})).content


def test_prefix_byte_identical_and_block_only_difference(env, agents):
    a, b = agents
    assert b.system_prompt.startswith(a.system_prompt)
    suffix = b.system_prompt[len(a.system_prompt):]
    assert suffix == "\n\n" + STATE_BLOCK.format(tool_name="get_vehicle_state", state=state_tool_output(env))


def test_block_body_equals_tool_output(env, agents):
    _, b = agents
    assert state_tool_output(env) in b.render_state_block()


def test_condition_a_has_no_block(agents):
    a, _ = agents
    assert "<vehicle_state>" not in a.system_prompt


def test_same_tools_and_policy(agents):
    a, b = agents
    assert [t.name for t in a.tools] == [t.name for t in b.tools]
    assert a.domain_policy == b.domain_policy
    assert a.llm == b.llm and a.llm_args == b.llm_args


def test_block_reflects_initialization_actions(env, agents):
    _, b = agents
    env.run_env_function_calls([EnvFunctionCall(env_type="assistant", func_name="set_vehicle_state",
                                                arguments={"vehicle": {"gear_selection": "D", "perf_vehicle_speed_kmh": 60.0}})])
    block = b.render_state_block()
    assert '"gear_selection": "D"' in block and '"perf_vehicle_speed_kmh": 60.0' in block


def test_block_is_live_per_turn(env, agents, monkeypatch):
    _, b = agents
    captured = []

    def fake_generate(model, tools, messages, call_name, **kwargs):
        captured.append(messages[0].content)
        return AssistantMessage(role="assistant", content="ok", cost=0.0)

    monkeypatch.setattr(ca, "generate", fake_generate)
    state = b.get_init_state()
    b.generate_next_message(UserMessage(role="user", content="hi"), state)
    env.get_response(ToolCall(id="1", name="set_volume", arguments={"level": 27}))
    b.generate_next_message(UserMessage(role="user", content="louder"), state)
    assert '"volume": 12' in captured[0]
    assert '"volume": 27' in captured[1]
    assert captured[1].startswith(state.system_messages[0].content[: len("<instructions>")])


def test_noop_agent_never_calls_tools(env):
    agent = NoOpAgent(tools=env.get_tools(), domain_policy=env.get_policy())
    state = agent.get_init_state()
    msg, state = agent.generate_next_message(UserMessage(role="user", content="unlock the doors"), state)
    assert msg.tool_calls is None and msg.content
