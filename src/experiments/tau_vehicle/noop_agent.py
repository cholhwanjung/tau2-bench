"""Do-nothing agent: never calls a tool, always replies with the same sentence.

Used to measure the fraction of tasks that pass with an unchanged environment
(Benchmark Checklist / ABC "trivial agent" check). No LLM calls.
"""

from typing import List, Optional

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.environment.tool import Tool

NOOP_REPLY = "Sorry, I can't help with that right now."


class NoOpAgent(HalfDuplexAgent[LLMAgentState]):
    def __init__(self, tools: List[Tool], domain_policy: str, **kwargs):
        super().__init__(tools=tools, domain_policy=domain_policy)

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> LLMAgentState:
        return LLMAgentState(system_messages=[], messages=list(message_history or []))

    def generate_next_message(self, message, state: LLMAgentState):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        reply = AssistantMessage(role="assistant", content=NOOP_REPLY, cost=0.0)
        state.messages.append(reply)
        return reply, state

    def set_seed(self, seed: int):
        pass


def create_noop_agent(tools, domain_policy, **kwargs):
    return NoOpAgent(tools=tools, domain_policy=domain_policy)
