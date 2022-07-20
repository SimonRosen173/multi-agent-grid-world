# Some rendering code adapted from https://github.com/raillab/composition/blob/master/gym_repoman

from collections import defaultdict, OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, IO, Set

import os
import itertools

from utils import loadmap

import numpy as np

import gym
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame

import pygame
from pygame import sprite


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


PATH_SEP = os.path.sep
BASE_PATH = os.path.split(os.path.abspath(__file__))[0]
ASSETS_PATH = os.path.join(BASE_PATH, 'assets')


def _load_image(name):
    img_path = os.path.join(ASSETS_PATH, name)
    try:
        image = pygame.image.load(img_path)
    except pygame.error:
        print('Cannot load image:', img_path)
        raise SystemExit()
    image = image.convert_alpha()
    return image


def _calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]


###########
# SPRITES #
###########
# TODO: Test
class _GoalSprite(sprite.Sprite):
    # TODO: Add images
    _GOAL_IMAGES = {
        "desirable": "blue_square.png",
        "undesirable": "beige_square.png",
    }

    def __init__(self, sprite_size: int, is_desirable: bool, pos: Tuple[int, int] = None):
        is_desirable_str = "desirable" if is_desirable else "undesirable"
        self.name = f"goal_{is_desirable_str}"
        self._is_desirable = is_desirable
        self._sprite_size = sprite_size
        sprite.Sprite.__init__(self)

        image = _load_image(self._GOAL_IMAGES[is_desirable_str])
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = pos
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)

    def reset(self, position: Tuple[int, int]):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)


class _AgentSprite(sprite.Sprite):
    _IMAGE_BASE_NAME = "character"  # TODO
    _COLORS = ["red", "blue"]

    def __init__(self, sprite_size: int, agent_no: int, start_pos: Tuple[int,int]):
        self.name = f"agent_{agent_no}"
        self._sprite_size = sprite_size
        self._agent_no = agent_no

        sprite.Sprite.__init__(self)
        assert agent_no < len(self._COLORS), f"agent_no too high (agent_no={agent_no}>={len(self._COLORS)})"
        img_name = f"{self._IMAGE_BASE_NAME}_{self._COLORS[agent_no]}.png"
        image = _load_image(img_name)
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()

        self.position = None
        self.update_pos(start_pos)

    def update_pos(self, pos):
        self.position = pos
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)

    def reset(self, position: Tuple[int, int]):
        self.update_pos(position)
        # self.position = position
        # self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def step(self, move: Tuple[int, int]):
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)


###############
# Environment #
###############
class GridWorld(gym.Env):
    _DEFAULT_REWARDS = {}
    _DEFAULT_RENDERING_CONFIG = {
        "sprite_size": 40,
        # "screen_size": (400, 400)
    }
    _DEFAULT_DYNAMICS_CONFIG = {}

    _ASSET_IMAGES = {
        "wall": "wall.png",
        "ground": "ground.png"
    }

    _GRID_SPRITE_KEYS = {
        0: "ground",
        1: "wall"
    }

    def __init__(self,
                 n_agents: int, grid: Union[str, IO, np.ndarray],
                 goals: Set[Tuple[int, int]], desirable_joint_goals: Set[Tuple[Tuple[int, int], ...]],
                 joint_start_state: List[Tuple[int, int]],
                 grid_input_type: str = "arr",  # ["arr","str","file_path", "map_name"]
                 rewards: Dict[str, float] = {},
                 slip_prob: float = 0.0,
                 actions_type="cardinal",  # ["cardinal","turn"]
                 observations_type="joint_state",  # ["joint_state","rgb"],
                 is_rendering=False,
                 rendering_config={},
                 dynamics_config: Dict = {},
                 ):

        assert n_agents > 0, f"n_agents must be greater than 0 (n_agents was {n_agents})"
        self._n_agents: int = n_agents

        # Rewards
        # GridWorld._validate_rewards(rewards)
        self._rewards = _copy_to_dict(rewards, GridWorld._DEFAULT_REWARDS)

        # ACTIONS
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

        self._actions_map_e2s = {}  # Maps action enum to action in action_space - enum to(2) space
        self._actions_map_s2e = {} # Maps action in action_space to action enum - space to(2) enum
        for i, val in enumerate(self._valid_actions):
            self._actions_map_e2s[val] = i
            self._actions_map_s2e[i] = val
        # action_space = spaces.Discrete(self._n_actions)
        self.joint_action_space = spaces.Tuple([spaces.Discrete(self._n_actions) for _ in range(self._n_agents)])

        # OBSERVATIONS
        self._observations_type = observations_type

        # Grid
        self._grid: np.ndarray = self._load_grid(grid, grid_input_type)
        self._width = self._grid.shape[0]
        self._height = self._grid.shape[1]

        # Agents
        # Data oriented model chosen for agents for efficiency and performance
        self._agents_direction = [0 for _ in range(n_agents)]
        self._agents_pos = joint_start_state
        self._joint_start_state = joint_start_state
        # self._agents_x = [0 for _ in range(n_agents)]
        # self._agents_y = [0 for _ in range(n_agents)]

        # Goals
        self._goals = goals
        self._joint_goals = set(itertools.product(*[list(goals) for _ in range(n_agents)]))
        self._desirable_joint_goals: Set[Tuple[Tuple[int, int], ...]] = desirable_joint_goals

        self._desirable_goals: Set[Tuple[int, int]] = set()  # This is for rendering
        for joint_goal in self._desirable_joint_goals:
            for goal in joint_goal:
                self._desirable_goals.add(goal)
        self._undesirable_goals = self._goals.difference(self._desirable_goals)

        # Dynamics
        # Indicates if a number in the grid can be collided with based off 'no >= _collision_threshold'
        self._collision_threshold = 1
        self._dynamics_config = _copy_to_dict(dynamics_config, GridWorld._DEFAULT_DYNAMICS_CONFIG)

        # Rendering
        self.viewer = None
        self._is_rendering_init = False
        self._rendering_config = _copy_to_dict(rendering_config, self._DEFAULT_RENDERING_CONFIG)

        if is_rendering or observations_type == "rgb":
            self._init_rendering()

    def _init_rendering(self):
        self._is_rendering_init = True
        # Set up PyGame stuff
        assert "sprite_size" in self._rendering_config
        # Calc window size
        sprite_size = self._rendering_config["sprite_size"]
        window_size = (self._width * sprite_size, self._height * sprite_size)
        self._rendering_config["window_size"] = window_size

        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"  # Set window to open at top left of screen

        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1, 1))

        self._bestdepth = pygame.display.mode_ok(window_size, 0, 32)
        self._surface = pygame.Surface(window_size, 0, self._bestdepth)
        self._background = pygame.Surface(window_size)
        self._clock = pygame.time.Clock()

        self._make_render_grid()

        self._goals_group = pygame.sprite.Group()
        self._agents_group = pygame.sprite.Group()
        self._render_group = pygame.sprite.Group()
        # self.player = _AgentSprite(sprite_size, 0)
        # self.render_group.add(self.player)

        # Goals
        for goal in self._desirable_goals:
            self._goals_group.add(_GoalSprite(sprite_size, True, goal))
        for goal in self._undesirable_goals:
            self._goals_group.add(_GoalSprite(sprite_size, False, goal))
        self._render_group.add(self._goals_group.sprites())

        # Agents
        for agent_no in range(self._n_agents):
            self._agents_group.add(_AgentSprite(sprite_size, agent_no,
                                                self._agents_pos[agent_no]))
        self._render_group.add(self._agents_group.sprites())

    @staticmethod
    def _validate_rewards(rewards):
        raise NotImplementedError

    # Validate params set in __init__
    def _validate_init(self):
        pass

    @staticmethod
    def _load_grid(grid, grid_input_type: str):
        _grid = None
        if type(grid) == np.ndarray:
            _grid = grid
        elif type(grid) == list and type(grid[0]) == list:
            _grid = np.asarray(grid, dtype=int)
        elif grid_input_type == "file_path" or grid_input_type == "map_name":
            _grid = loadmap.load(grid, grid_input_type)

        return _grid

    # PYGAME RENDERING
    def _make_render_grid(self):
        grid = self._grid
        sprite_size = self._rendering_config["sprite_size"]
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                curr_tile = self._GRID_SPRITE_KEYS[grid[row][col]]
                if curr_tile == "ground":
                    image_name = self._ASSET_IMAGES["ground"]
                elif curr_tile == "wall":
                    image_name = self._ASSET_IMAGES["wall"]
                else:
                    raise NotImplementedError
                image = _load_image(image_name)
                image = pygame.transform.scale(image, (sprite_size, sprite_size))
                position = _calculate_topleft_position((row, col), sprite_size)
                self._background.blit(image, position)

    def _draw_screen(self, surface):
        surface.blit(self._background, (0, 0))
        self._render_group.draw(surface)
        surface_array = pygame.surfarray.array3d(surface)
        observation = np.copy(surface_array).swapaxes(0, 1)
        del surface_array
        return observation

    def close(self):
        pygame.display.quit()

    # HELPER
    # Redo to make static so I can use njit
    def _take_joint_action(self, joint_action: List[int]):
        grid = self._grid

        def in_bounds(pos: Tuple[int,int]):
            y, x = pos
            return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]

        actions_map_s2e = self._actions_map_s2e
        if self._actions_type == "cardinal":
            curr_joint_pos = self._agents_pos
            next_joint_pos = []
            for action, curr_pos in zip(joint_action, curr_joint_pos):
                y, x = curr_pos
                # cand_pos - candidate pos
                if actions_map_s2e[action] == Action.NORTH: # y, x
                    cand_pos = (y-1, x)
                elif actions_map_s2e[action] == Action.EAST:
                    cand_pos = (y, x+1)
                elif actions_map_s2e[action] == Action.SOUTH:
                    cand_pos = (y+1, x)
                elif actions_map_s2e[action] == Action.WEST:
                    cand_pos = (y, x-1)
                else:
                    cand_pos = None

                if in_bounds(cand_pos) and grid[cand_pos] == 0:
                    next_pos = cand_pos
                else:
                    next_pos = curr_pos

                next_joint_pos.append(next_pos)
        else:
            raise NotImplementedError

    # GYM
    def reset(self, **kwargs):
        pass

    def step(self, action: ActType) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        # If RGB -> return RGB array
        # If joint state -> return list of states
        if self._observations_type == "joint_state":
            pass
        elif self._observations_type == "rgb":
            pass
        else:
            raise NotImplementedError

    def render(self, mode="human", close=False) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if not self._is_rendering_init:
            self._init_rendering()

        if close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = pygame.display.set_mode(self._rendering_config["window_size"], 0, self._bestdepth)

        self._clock.tick(10 if mode != 'human' else 2)
        arr = self._draw_screen(self.viewer)
        pygame.display.flip()
        return arr


def test():
    TL = (2, 2)  # y,x
    BL = (10, 2)
    TR = (2, 10)
    BR = (10, 10)
    goals = {TL, BL, TR, BR}
    desirable_joint_goals = {(TL, TR), (TR, TL), (TR, TR), (TL, TL)}
    joint_start_state = [(1, 1), (11, 11)]

    env = GridWorld(2, "corridors", goals=goals, desirable_joint_goals=desirable_joint_goals,
                    joint_start_state=joint_start_state,
                    grid_input_type="map_name", is_rendering=True)
    # env = GridWorld(2, "corridors", {(1, 1)}, {(1, 1)}, grid_input_type="map_name", is_rendering=True)
    print(env.render())
    print(env.joint_action_space.sample())
    input()


if __name__ == "__main__":
    test()

    # pygame.time.delay(2000)

