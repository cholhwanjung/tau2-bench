"""Condition B agent: stock LLMAgent + live `<vehicle_state>` block.

Isolation guarantees (checked by tests/test_context_agent.py):
1. B.system_prompt == A.system_prompt + "\n\n" + block  (byte-identical prefix)
2. block body == exactly what the `get_vehicle_state` tool returns
3. same tools, model, policy; the block is re-rendered on every agent turn
"""

from typing import List, Optional

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

STATE_TOOL_NAME = "get_vehicle_state"

STATE_BLOCK = """<vehicle_state>
Current vehicle state (auto-updated every turn; identical to the output of {tool_name}):
{state}
</vehicle_state>"""


class VehicleContextAgent(LLMAgent):
    """LLMAgent whose system prompt ends with the live vehicle state."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        state_tools = [t for t in tools if t.name == STATE_TOOL_NAME]
        if len(state_tools) != 1:
            raise ValueError(f"Expected exactly one {STATE_TOOL_NAME} tool, got {len(state_tools)}")
        self._state_tool = state_tools[0]

    def render_state_block(self) -> str:
        # Tool objects are bound to the live toolkit/DB, so calling it here reads
        # the current state without going through the orchestrator.
        state = Environment.to_json_str(self._state_tool())
        return STATE_BLOCK.format(tool_name=STATE_TOOL_NAME, state=state)

    @property
    def system_prompt(self) -> str:
        return super().system_prompt + "\n\n" + self.render_state_block()

    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        # Re-render the system prompt every turn so the block is live.
        state.system_messages = [SystemMessage(role="system", content=self.system_prompt)]
        messages = state.system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )


def create_vehicle_context_agent(tools, domain_policy, **kwargs):
    """Factory for registry.register_agent_factory(..., "llm_agent_vehicle_ctx")."""
    return VehicleContextAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
