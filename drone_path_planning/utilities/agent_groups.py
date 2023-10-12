import dataclasses
from typing import Any
from typing import Dict
from typing import Set

from drone_path_planning.agents import DeepQNetworkAgent
from drone_path_planning.utilities.training_helpers import ReplayBuffer


@dataclasses.dataclass
class MultiAgentGroup:
    agent: DeepQNetworkAgent
    agent_compile_kwargs: Dict[str, Any]
    agent_ids: Set[str]


@dataclasses.dataclass
class MultiAgentTrainingGroup(MultiAgentGroup):
    replay_buffer: ReplayBuffer
    replay_buffer_size: int


@dataclasses.dataclass
class VariableNumberAgentGroup:
    agent: DeepQNetworkAgent
    agent_compile_kwargs: Dict[str, Any]


@dataclasses.dataclass
class VariableNumberAgentTrainingGroup(VariableNumberAgentGroup):
    replay_buffer: ReplayBuffer
    replay_buffer_size: int
