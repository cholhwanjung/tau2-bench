"""Ablation agents.

D = push without pull: VehicleContextAgent whose tool list excludes
`get_vehicle_state` (the block is still rendered from the environment's tool).
"""

from experiments.tau_vehicle.context_agent import STATE_TOOL_NAME, VehicleContextAgent


class VehicleContextNoReadAgent(VehicleContextAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        # keep the bound state tool for rendering, hide it from the LLM
        self.tools = [t for t in tools if t.name != STATE_TOOL_NAME]


def create_vehicle_context_noread_agent(tools, domain_policy, **kwargs):
    return VehicleContextNoReadAgent(
        tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args")
    )
