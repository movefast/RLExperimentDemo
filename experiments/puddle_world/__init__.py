from dataclasses import dataclass

from agents.planning.base_agent import QLearningAgent
from agents.planning.forward_agent import BackwardAgent, ForwardAgent
from agents.planning.per_agent import PERAgent
from agents.planning.qr_agent import QRAgent
from common.agent_helpers import PriorityTYPE

# from gym.envs.registration import register
# register(
#     id='PuddleWorld-v0',
#     entry_point='environments.puddle_world:PuddleEnv',
# )

AGENT_REG = {
    "QLearning": QLearningAgent,
    "PER": PERAgent,
    "Forward": ForwardAgent,
    "Backward": BackwardAgent,
    "QR": QRAgent,
}


@dataclass
class DefaultConfig:
    # default for tc
    init: float = 0
    step_size: float = 0.00125
    batch_size: int = 1
    buffer_size: int = 10000
    bias: bool = False
    beta: float = 0.0
    emphatic: bool = False
    ptype: bool = PriorityTYPE.PER
    loss_func: str = "Huber"
    is_episodic_bound: bool = True


@dataclass
class DefaultPERConfig(DefaultConfig):
    per_alpha: float = 1
    buffer_beta: float = 0.4
    beta_increment: float = (1-0.4)/10000
    min_weight: float = 1e-5
    importance_sampling: bool = False


@dataclass
class DefaultForwardConfig(DefaultConfig):
    replay_across_iter: bool = False


@dataclass
class DefaultBackwardConfig(DefaultConfig):
    lam: float = 1
    agent_type: int = 6
    esp: float = None
    max_replay_count: int = None
    replay_across_iter: bool = False
    skip_last_mom:bool = True


AGENT_DEFAULT_PARAMS = {
    "QLearning": DefaultConfig(),
    "Forward": DefaultForwardConfig(),
    "Backward": DefaultBackwardConfig(),
    "PER": DefaultPERConfig(),
    "QR": DefaultConfig(),
}
