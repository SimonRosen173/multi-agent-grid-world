from collections import defaultdict, OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, IO, Set

import itertools

from utils import loadmap

import numpy as np

import gym
from gym.core import ActType, ObsType, RenderFrame


#########
# ENUMS #
#########
class Action(Enum):
    NOOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_UP = 4
    NORTH = 5
    EAST = 6
    SOUTH = 7
    WEST = 8


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


###########
# HISTORY #
###########
class History:
    def __init__(self):
        pass


##################
# HELPER METHODS #
##################
# There is probs a method that does this already :P
def _copy_to_dict(from_dict, to_dict):
    for key in from_dict.keys():
        to_dict[key] = from_dict[key]
    return to_dict


###############
# Environment #
###############
class GridWorld(gym.Env):
    _DEFAULT_REWARDS = {}
    _DEFAULT_RENDERING_CONFIG = {}
    _DEFAULT_DYNAMICS_CONFIG = {}

    def __init__(self,
                 n_agents: int, grid: Union[str, IO, np.ndarray], grid_input_type: str,
                 goals: Set[Tuple[int, int]], desirable_joint_goals: Set[Tuple[int, int]],
                 rewards: Dict[float] = {},
                 slip_prob: float = 0.0,
                 actions_type="cardinal",  # ["cardinal","turn"]
                 observations_type="joint_state",  # ["joint_state","rgb"],
                 is_rendering=False,
                 render_config={},
                 dynamics_config: Dict = {},
                 ):
        assert n_agents > 0, f"n_agents must be greater than 0 (n_agents was {n_agents})"
        self._n_agents: int = n_agents

        # Rewards
        GridWorld._validate_rewards(rewards)
        self._rewards = _copy_to_dict(rewards, GridWorld._DEFAULT_REWARDS)

        # Actions
        self._slip_prob = slip_prob
        self._actions_type = actions_type
        self._valid_actions = [Action.NOOP, Action.PICK_UP]
        if actions_type == "cardinal":
            self._move_actions = [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST]
        elif actions_type == "turn":
            self._move_actions = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]
        else:
            raise NotImplementedError(f"actions_type == {actions_type} is not supported")
        self._valid_actions.extend(self._move_actions)
        self._n_actions = len(self._valid_actions)

        # Grid
        self._grid: np.ndarray = self._make_grid(grid, grid_input_type)
        self._width = self._grid.shape[0]
        self._height = self._grid.shape[1]

        # Agents
        # Data oriented model chosen for agents for efficiency and performance
        self._agents_direction = [0 for _ in range(n_agents)]
        self._agents_x = [0 for _ in range(n_agents)]
        self._agents_y = [0 for _ in range(n_agents)]

        # Goals
        self._goals = goals
        self._joint_goals = set(itertools.product(*[list(goals) for _ in range(n_agents)]))
        self._desirable_joint_goals = desirable_joint_goals

        # Dynamics
        # Indicates if a number in the grid can be collided with based off 'no >= _collision_threshold'
        self._collision_threshold = 1
        self._dynamics_config = _copy_to_dict(dynamics_config, GridWorld._DEFAULT_DYNAMICS_CONFIG)

        # Rendering
        if is_rendering or observations_type == "rgb":
            # Set up PyGame stuff
            pass

    @staticmethod
    def _validate_rewards(rewards):
        raise NotImplementedError

    # Validate params set in __init__
    def _validate_init(self):
        pass

    @staticmethod
    def _make_grid(grid, grid_input_type: str):
        _grid = None
        if type(grid) == np.ndarray:
            _grid = grid
        elif type(grid) == list and type(grid[0]) == list:
            _grid = np.asarray(grid, dtype=int)
        elif grid_input_type == "file_name" or grid_input_type == "map_name":
            _grid = loadmap.load(grid, grid_input_type)

        return _grid

    def reset(self, **kwargs):
        pass

    def step(self, action: ActType) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        # If RGB -> return Box
        # If joint state -> Return Tuple(Discrete(n_actions), Discrete(n_actions), ...)
        pass

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


if __name__ == "__main__":
    pass
