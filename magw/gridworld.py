# Some rendering code adapted from https://github.com/raillab/composition/blob/master/gym_repoman

from typing import List, Tuple, Optional, Dict, Union, IO, Set
import copy

import os
import sys

import numpy as np

import networkx as nx

import gym
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame


from magw.utils.general import copy_to_dict
from magw.utils import loadmap
from magw.history import EnvHistory
from magw.enums import Action, Direction
from magw.sprites import GoalSprite, AgentSprite, load_image, calculate_topleft_position

import pygame


###############
# Environment #
###############
class GridWorld(gym.Env):
    _DEFAULT_REWARDS = {
        "step": -0.02,
        "wait": -0.01,
        "wait_at_goal": -0.001,
        "collision": -0.01,
        "desirable_goal": 10,
        "undesirable_goal": -10
    }
    _DEFAULT_RENDERING_CONFIG = {
        "sprite_size": 40,
        # "screen_size": (400, 400)
    }
    _DEFAULT_DYNAMICS_CONFIG = {
        "collisions_enabled": True,
        "collisions_at_goals": True,
        "slip_prob": 0,
    }

    _ASSET_IMAGES = {
        "wall": "wall.png",
        "ground": "ground.png"
    }

    _GRID_SPRITE_KEYS = {
        0: "ground",
        1: "wall"
    }

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    def __init__(self,
                 n_agents: int, grid: Union[str, IO, np.ndarray],
                 goals: Set[Tuple[int, int]],
                 desirable_joint_goals: Set[Tuple[Tuple[int, int], ...]],
                 joint_start_state: List[Tuple[int, int]],
                 grid_input_type: str = "arr",  # ["arr","str","file_path", "map_name"]
                 rewards_config: Dict[str, float] = {},
                 slip_prob: float = 0.0,
                 actions_type="cardinal",  # ["cardinal","turn"]
                 observations_type="joint_state",  # ["joint_state","rgb"],
                 flatten_state=False,  # Flatten state when returned
                 is_rendering=False,
                 render_mode="rgb_array",
                 rendering_config={},
                 dynamics_config: Dict = {},
                 logging_config: Optional[Dict] = None,
                 ):
        _locals = locals()

        exclude_keys = {"self", "_locals", "exclude_keys"}
        _locals = {key: value for (key, value) in _locals.items() if key not in exclude_keys}

        self._init_args = copy.deepcopy(_locals)

        # self._init_args = copy.deepcopy(locals())
        # del self._init_args["self"]

        # Env variable
        self.render_mode = render_mode

        assert n_agents > 0, f"n_agents must be greater than 0 (n_agents was {n_agents})"
        self._n_agents: int = n_agents

        self._flatten_state = flatten_state

        self.episode_no = 0

        # Rewards
        # GridWorld._validate_rewards_config(rewards_config)
        self._rewards_config = copy_to_dict(rewards_config, GridWorld._DEFAULT_REWARDS)

        # ACTIONS
        self._slip_prob = slip_prob
        self._actions_type = actions_type
        self._valid_actions = [Action.WAIT]  # , Action.PICK_UP] For now WAIT acts as pickup
        if actions_type == "cardinal":
            self._move_actions = [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST]
        elif actions_type == "turn":
            self._move_actions = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]
        else:
            raise NotImplementedError(f"actions_type == {actions_type} is not supported")
        self._valid_actions.extend(self._move_actions)
        self._n_actions = len(self._valid_actions)

        self._actions_map_e2s = {}  # Maps action enum to action in action_space - enum to(2) space
        self._actions_map_s2e = {}  # Maps action in action_space to action enum - space to(2) enum
        for i, val in enumerate(self._valid_actions):
            self._actions_map_e2s[val] = i
            self._actions_map_s2e[i] = val

        # OBSERVATIONS
        self._observations_type = observations_type

        # GRID
        self._grid: np.ndarray = self._load_grid(grid, grid_input_type)
        self._width = self._grid.shape[0]
        self._height = self._grid.shape[1]
        # Find valid states
        # NOTE: Only counts 0 as valid state
        self._valid_states: List[Tuple[int, int]] = list(map(tuple, np.argwhere(self._grid == 0)))
        self.n_valid_states: int = len(self._valid_states)

        # AGENTS
        # Data oriented model chosen for agents for efficiency and performance
        self._agents_direction = [0 for _ in range(n_agents)]
        self._joint_pos: List[Tuple[int, int]] = joint_start_state
        self._joint_start_state: List[Tuple[int, int]] = joint_start_state

        # Goals
        self._goals: Set[Tuple[int, int]] = goals
        # self._joint_goals = set(itertools.product(*[list(goals) for _ in range(n_agents)]))
        self._desirable_joint_goals: Set[Tuple[Tuple[int, int], ...]] = desirable_joint_goals

        self._desirable_goals: Set[Tuple[int, int]] = set()  # This is for rendering
        for joint_goal in self._desirable_joint_goals:
            for goal in joint_goal:
                self._desirable_goals.add(goal)
        self._undesirable_goals = self._goals.difference(self._desirable_goals)

        # Dynamics
        # Indicates if a number in the grid can be collided with based off 'no >= _collision_threshold'
        self._collision_threshold = 1
        self._dynamics_config = copy_to_dict(dynamics_config, GridWorld._DEFAULT_DYNAMICS_CONFIG)
        if not self._dynamics_config["collisions_at_goals"]:
            print("[WARNING] collisions at goals is not fully working")

        # HISTORY
        self.env_history = EnvHistory(n_agents, actions_type, logging_config,
                                      self._joint_start_state,self.episode_no)

        # Rendering
        self.viewer = None
        self._is_rendering_init = False
        self._rendering_config = copy_to_dict(rendering_config, self._DEFAULT_RENDERING_CONFIG)

        if is_rendering or observations_type == "rgb":
            self._init_rendering()

        # GYM SPACES
        # Action Space
        self.action_space = spaces.Tuple([spaces.Discrete(self._n_actions) for _ in range(self._n_agents)])

        # Observation Space
        if self._observations_type == "joint_state":
            if self._flatten_state:
                self.observation_space = spaces.Discrete(self._n_agents)
            else:
                self.observation_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self._n_agents)])
        elif self._observations_type == "rgb":
            raise NotImplementedError("RGB observation space has not been implemented")

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
        self._render_group = pygame.sprite.RenderPlain()
        # self.player = AgentSprite(sprite_size, 0)
        # self.render_group.add(self.player)

        # Goals
        for goal in self._desirable_goals:
            self._goals_group.add(GoalSprite(sprite_size, True, goal))
        for goal in self._undesirable_goals:
            self._goals_group.add(GoalSprite(sprite_size, False, goal))
        self._render_group.add(self._goals_group.sprites())

        # Agents
        for agent_no in range(self._n_agents):
            self._agents_group.add(AgentSprite(sprite_size, agent_no,
                                                self._joint_pos[agent_no]))
        self._render_group.add(self._agents_group.sprites())

    @property
    def n_agents(self):
        return self._n_agents

    @property
    def flatten_state(self):
        return self._flatten_state

    def get_config(self):
        config_dict = copy.deepcopy(self._init_args)
        # Config cannot contain sets since it needs to be JSON serializable
        config_dict["desirable_joint_goals"] = list(config_dict["desirable_joint_goals"])
        config_dict["goals"] = list(config_dict["goals"])

        config_dict["rewards_config"] = self._rewards_config
        config_dict["rendering_config"] = self._rendering_config
        config_dict["dynamics_config"] = self._dynamics_config
        config_dict["logging_config"] = self.env_history.logging_config
        return config_dict

    @staticmethod
    def _validate_rewards_config(rewards_config):
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

    # HELPER METHODS
    def flatten_states(self, joint_state: List[Tuple[int, int]]) -> List[int]:
        return [int(np.ravel_multi_index(state, self._grid.shape)) for state in joint_state]

    def unflatten_states(self, joint_state: List[int]) -> List[Tuple[int, int]]:
        return [tuple(np.unravel_index(state, self._grid.shape)) for state in joint_state]

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
                image = load_image(image_name)
                image = pygame.transform.scale(image, (sprite_size, sprite_size))
                position = calculate_topleft_position((row, col), sprite_size)
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

    # TODO: Add NJIT
    # @staticmethod
    # def _check_agent_collisions(prev_joint_pos, cand_next_joint_pos, info,
    #                             goals: Set[Tuple[int, int]], collisions_at_goals: bool = False):
    #     if collisions_at_goals:
    #         goals = set()  # Empty collisions set because goals info is no longer useful
    #
    #     n_agents = len(cand_next_joint_pos)
    #     reward = 0
    #     next_joint_pos = []
    #     collisions = [0 for _ in range(n_agents)]
    #     # 1. End in same cell collision
    #     # Collisions should be infrequent, so need fast check to see if any happened
    #     # Check if collisions in cand_next_joint_pos
    #     if len(cand_next_joint_pos) != len(set(cand_next_joint_pos)):  # O(n)
    #         # Now check where the collisions happened
    #         for i in range(len(cand_next_joint_pos)):
    #             for j in range(i, len(cand_next_joint_pos)):
    #                 if cand_next_joint_pos[i] == cand_next_joint_pos[j]:
    #                     # Collisions
    #                     if cand_next_joint_pos[i] not in goals:
    #                         collisions[i] = 1
    #                         collisions[j] = 1
    #                         # Maybe something to do with reward?
    #         pass
    #
    #     return next_joint_pos, reward, info

    @staticmethod
    def _check_agent_collisions(
            prev_joint_pos: List[Tuple[int,int]],
            cand_next_joint_pos: List[Tuple[int,int]],
            goals: Set[Tuple[int, int]],
            collisions_at_goals: bool = False,
    ):
        info = ""

        n_agents = len(prev_joint_pos)
        nodes_set = set(prev_joint_pos + cand_next_joint_pos)

        dg = nx.DiGraph()
        dg.add_nodes_from(nodes_set)
        edges_list = list(zip(prev_joint_pos, cand_next_joint_pos))
        dg.add_edges_from(edges_list)

        # Set attributes
        # node attributes
        node_attributes = {node: {"is_goal": node in goals} for node in dg.nodes()}
        nx.set_node_attributes(dg, node_attributes)

        # edge attributes
        edge_agent_id = [i for i in range(n_agents)]
        edge_is_stationary = [(True if x[0] == x[1] else False) for x in edges_list]
        edge_attributes = {edges_list[i]: {"agent_id": i, "is_stationary": edge_is_stationary[i]}
                           for i in range(len(edges_list))}
        nx.set_edge_attributes(dg, edge_attributes)

        # Collisions
        n_removed_edges = 0
        problem_agents = set()

        # Pass through collisions
        edges = [edge for edge in dg.edges()]
        edges_seen = {edge: False for edge in edges}
        for edge in edges:
            if not edges_seen[edge]:
                rev_edge = (edge[1], edge[0])
                if rev_edge in dg.edges():
                    edges_seen[rev_edge] = True
                    edge_attr = dg[edge[0]][edge[1]]
                    rev_edge_attr = dg[rev_edge[0]][rev_edge[1]]
                    problem_agents.add(edge_attr["agent_id"])
                    problem_agents.add(rev_edge_attr["agent_id"])
                    n_removed_edges += 2

                    dg.remove_edge(*edge)
                    dg.add_edge(edge[0], edge[0], **edge_attr)
                    dg.remove_edge(*rev_edge)
                    dg.add_edge(rev_edge[0], rev_edge[0], **rev_edge_attr)

        # Same cell collisions
        in_edges_nodes = [(node, list(dg.in_edges(node))) for node in dg.nodes()]
        problem_nodes_edges = list(filter(lambda x: True if len(x[1]) > 1 else False, in_edges_nodes))
        problem_nodes = [el[0] for el in problem_nodes_edges]

        while len(problem_nodes) > 0:
            problem_node = problem_nodes.pop(0)
            # False if node is goal and collisions do not occur at goals
            if not (problem_node in goals and not collisions_at_goals):
                problem_node_edges = list(dg.in_edges(problem_node))
                non_stationary_edges = list(filter(lambda x: False if x[0] == x[1] else True,
                                                   problem_node_edges))
                for edge in non_stationary_edges:
                    # print(edge)
                    n_removed_edges += 1
                    edge_attr = dg[edge[0]][edge[1]]
                    problem_agents.add(edge_attr["agent_id"])

                    new_edge = (edge[0], edge[0])  # Stationary edge
                    dg.add_edge(*new_edge, **edge_attr)
                    dg.remove_edge(*edge)
                    if len(dg.in_edges(new_edge[0])) > 1:
                        problem_nodes.append(new_edge[0])

        # print("No of edges removed: ", n_removed_edges)
        # print("Problem agents: ", list(problem_agents))

        # Compute next_joint_pos
        next_joint_pos = [None for _ in range(n_agents)]
        next_pos_agent = [(dg[edge[0]][edge[1]]["agent_id"], edge[1]) for edge in dg.edges()]

        for agent_id, pos in next_pos_agent:
            next_joint_pos[agent_id] = pos

        # Counting collisions based off the cand_pos had to be changed
        # If an agent waited and the other agent collided with it, I am counting that as one collision
        # while it counts as two collisions if both agents tried to move into each other
        n_collisions = 0
        for i in range(n_agents):
            if cand_next_joint_pos[i] != next_joint_pos[i]:
                n_collisions += 1
        # n_collisions = len(problem_agents)

        # print("next_joint_pos:", next_joint_pos)

        return next_joint_pos, n_collisions, info

    # HELPER
    # Redo to make static so I can use njit
    def _take_joint_action(self, joint_action: Union[List[int], List[Action]]):
        # TODO: Fix bug where next_joint_pos contains None values if agents start from same location
        #  and end at same location
        grid = self._grid
        rewards_config = self._rewards_config
        n_agents = self._n_agents

        def in_bounds(pos: Tuple[int,int]):
            _y, _x = pos
            return 0 <= _y < grid.shape[0] and 0 <= _x < grid.shape[1]

        reward = 0
        is_done = False
        info = ""

        # actions_map_s2e = self._actions_map_s2e
        if self._actions_type == "cardinal":
            curr_joint_pos = self._joint_pos
            cand_joint_pos = []
            action_set = {Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST}
            slip_map = {
                Action.NORTH: [Action.EAST, Action.WEST],
                Action.SOUTH: [Action.EAST, Action.WEST],
                Action.EAST: [Action.NORTH, Action.SOUTH],
                Action.WEST: [Action.NORTH, Action.SOUTH]
            }
            for action, curr_pos in zip(joint_action, curr_joint_pos):
                y, x = curr_pos
                # cand_pos - candidate pos
                # actions_map_s2e[action]
                cand_pos = None
                if action == Action.WAIT:
                    cand_pos = curr_pos
                    if curr_pos in self._goals:
                        reward += rewards_config["wait_at_goal"]
                    else:
                        reward += rewards_config["wait"]
                elif action == Action.PICK_UP:
                    raise NotImplementedError("Action.PICK_UP not supported. Use Action.WAIT instead.")
                elif action in action_set:
                    if np.random.rand() < self._dynamics_config["slip_prob"]:
                        action = np.random.choice(slip_map[action])

                    if action == Action.NORTH:  # y, x
                        cand_pos = (y-1, x)
                    elif action == Action.EAST:
                        cand_pos = (y, x+1)
                    elif action == Action.SOUTH:
                        cand_pos = (y+1, x)
                    elif action == Action.WEST:
                        cand_pos = (y, x-1)

                    reward += rewards_config["step"]
                else:
                    cand_pos = None

                # Check out of bounds and collisions between obstacles
                if in_bounds(cand_pos) and grid[cand_pos] == 0:
                    next_pos = cand_pos
                else:
                    next_pos = curr_pos

                cand_joint_pos.append(next_pos)

                # Check collisions between agents
            # self._joint_pos = next_joint_pos
        else:
            raise NotImplementedError

        joint_noop = [Action.WAIT for _ in range(n_agents)]
        if joint_action == joint_noop:
            next_joint_pos = self._joint_pos

            if tuple(next_joint_pos) in self._desirable_joint_goals:
                is_done = True
                reward = self._rewards_config["desirable_goal"]
                info += "[EPISODE TERMINATION] Episode terminated at desirable goal "
            else:
                # ANY joint goal - i.e. desirable or undesirable, i.e. any combination of goals
                is_at_joint_goal = True
                for pos in next_joint_pos:
                    if pos not in self._goals:
                        is_at_joint_goal = False
                        break
                if is_at_joint_goal:
                    is_done = True
                    reward = self._rewards_config["undesirable_goal"]
                    info += "[EPISODE TERMINATION] Episode terminated at undesirable goal "
        else:
            collisions_at_goals = self._dynamics_config["collisions_at_goals"]
            next_joint_pos, n_collisions, collision_info = \
                self._check_agent_collisions(prev_joint_pos=self._joint_pos,
                                             cand_next_joint_pos=cand_joint_pos,
                                             goals=self._goals, collisions_at_goals=collisions_at_goals)

            if n_collisions > 0:
                info += f"[COLLISION] {n_collisions} collisions occurred."

            reward += n_collisions * self._rewards_config["collision"]
            # next_joint_pos = cand_joint_pos  # Temp

        self._joint_pos = next_joint_pos

        return next_joint_pos, reward, is_done, info

    # GYM
    # noinspection PyMethodOverriding
    #
    def reset(self,
              joint_start_state: Optional[Union[List[Tuple[int, int]], List[int]]] = None,
              random_start: bool = False, increment_episode=True) -> Union[List[int], List[Tuple[int, int]]]:

        # if self._flatten_state and joint_start_state is not None:
        #     # TODO: test this
        #     assert type(joint_start_state[0]) == int, "If self._flatten_state=True then joint_start_state " \
        #                                                "must be of type List[int]"
        #     joint_start_state = self.unflatten_states(joint_start_state)

        if joint_start_state is not None:
            if type(joint_start_state[0]) == int:
                joint_start_state = self.unflatten_states(joint_start_state)
            elif type(joint_start_state[0]) == tuple:
                pass
            else:
                raise ValueError("joint_start_state is of wrong type")

        else:  #  joint_start_state is None
            if not random_start:
                joint_start_state = self._joint_start_state
            else:
                # Set joint_start_state to n_agents no of valid_states - i.e. each agent is at distinct valid
                # state
                valid_states_ind = list(range(len(self._valid_states)))
                joint_start_state_ind = list(np.random.choice(valid_states_ind, self._n_agents, replace=False))
                joint_start_state = [self._valid_states[ind] for ind in joint_start_state_ind]

        self._joint_pos = joint_start_state

        if increment_episode:
            self.episode_no += 1
        self.env_history.reset(self.episode_no, joint_start_state)

        if self._flatten_state:
            joint_start_state = self.flatten_states(joint_start_state)

        return joint_start_state

    def step(self, joint_action: Union[List[int], List[Action]], is_enum=False) -> \
            Tuple[ObsType, float, bool, dict]:
        joint_action = joint_action.copy()
        if not is_enum:
            # old_joint_action = joint_action
            for i, action in enumerate(joint_action):
                joint_action[i] = self._actions_map_s2e[action]
        # If RGB -> return RGB array
        # If joint state -> return list of states
        if self._observations_type == "joint_state":
            next_joint_pos, reward, is_done, info = self._take_joint_action(joint_action)
            self.env_history.step(joint_action=joint_action, next_joint_state=next_joint_pos,
                                  reward=reward, is_done=is_done, info=info)
            if self._flatten_state:
                next_joint_pos = self.flatten_states(next_joint_pos)

            return tuple(next_joint_pos), reward, is_done, {"desc": info}
            # return self._take_joint_action(joint_action)
        elif self._observations_type == "rgb":
            pass
        else:
            raise NotImplementedError

    def render(self, mode="human", close=False) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        # Probably need to actually do something with 'mode'
        if not self._is_rendering_init:
            self._init_rendering()

        if close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = pygame.display.set_mode(self._rendering_config["window_size"], 0, self._bestdepth)

        if mode == "human":
            self._clock.tick(2)

        # agent_sprites =
        # agent_sprites[0].update_pos((3,3))
        for agent, pos in zip(self._agents_group.sprites(), self._joint_pos):
            agent.update_pos(pos)

        arr = self._draw_screen(self.viewer)

        if mode == "human":
            pygame.display.flip()

        if self.env_history.logging_config["frames"]:
            self.env_history.log_frame(arr)

        return arr


def test():
    TL = (2, 2)  # y,x
    BL = (10, 2)
    TR = (2, 10)
    BR = (10, 10)
    goals = {TL, BL, TR, BR}
    desirable_joint_goals = {(TL, TR), (TR, TL), (TR, TR), (TL, TL)}
    joint_start_state = [(1, 1), (1, 3)]

    dynamics_config = {"slip_prob": 0.0}
    logging_config = {"frames": True}

    env = GridWorld(2, "corridors", goals=goals, desirable_joint_goals=desirable_joint_goals,
                    joint_start_state=joint_start_state, flatten_state=True,
                    grid_input_type="map_name", is_rendering=True,
                    dynamics_config=dynamics_config, logging_config=logging_config)

    print(env.action_space)
    print(env.observation_space)

    config = env.get_config()
    # print(env.get_config())

    interactive(env)


def interactive(env: GridWorld, video_path: Optional[str] = None):
    env.render()
    key_action_map = {
        # pygame.K_n: Action.NORTH,
        # pygame.K_s: Action.SOUTH,
        # pygame.K_e: Action.EAST,
        # pygame.K_w: Action.WEST,
        pygame.K_UP: Action.NORTH,
        pygame.K_DOWN: Action.SOUTH,
        pygame.K_RIGHT: Action.EAST,
        pygame.K_LEFT: Action.WEST,
        # pygame.K_p: Action.PICK_UP,
        pygame.K_x: Action.WAIT
    }

    key_buffer = []
    action_buffer = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                # print(f"Hello: {event.key}")
                # print(pygame.K_KP_ENTER)
                if event.key == 13:
                    if len(action_buffer) < env.n_agents:
                        print(f"Not enough actions chosen. Actions chosen so far = {len(action_buffer)}")
                    elif len(action_buffer) > env.n_agents:
                        print(f"Too many actions chosen. Actions chosen so far = {len(action_buffer)}")
                    else:
                        next_state, reward, is_done, info = env.step(action_buffer, is_enum=True)
                        print(f"next_state={next_state}, reward={reward}, is_done={is_done}, info={info}")
                        env.render()

                    action_buffer = []
                elif event.key in key_action_map.keys():
                    action_buffer.append(key_action_map[event.key])
                elif event.key == pygame.K_r:
                    joint_start_state = env.reset(random_start=True)
                    print(f"Env reset - joint_start_state = {joint_start_state}")
                    env.render()
                elif event.key == pygame.K_p:
                    env_hist = env.env_history
                    logging_config = env_hist.logging_config
                    # history_dict = env.env_history.to_dict()
                    print(f"#####################")
                    print(f"#      HISTORY      #")
                    print(f"#####################")
                    print(f"---------------------")
                    print(f"|       Stats       |")
                    print(f"---------------------")
                    print(f" episode no = {env_hist.episode_no}")
                    print(f" no steps = {env_hist.curr_step}")
                    print(f" cumulative reward = {env_hist.eps_return}")

                    if logging_config["joint_action"]:
                        print(f"---------------------")
                        print(f"|   Joint Actions    |")
                        print(f"---------------------")
                        print(f" Joint actions = {env_hist.joint_action_str_history}")

                    if logging_config["joint_state"]:
                        print(f"---------------------")
                        print(f"|    Joint States    |")
                        print(f"---------------------")
                        print(f" Joint states = {env_hist.joint_state_history}")

                    if logging_config["reward"]:
                        print(f"---------------------")
                        print(f"|      Rewards       |")
                        print(f"---------------------")
                        print(f" Rewards = {env_hist.reward_history}")

                    if logging_config["is_done"]:
                        print(f"---------------------")
                        print(f"|      Is Done       |")
                        print(f"---------------------")
                        print(f" Is done = {env_hist.is_done_history}")

                    if logging_config["info"]:
                        print(f"---------------------")
                        print(f"|       Infos        |")
                        print(f"---------------------")
                        print(f" Infos = {env_hist.info_history}")

                    print(f"#####################")

                    # print(env_hist.frame_history.keys())
                elif event.key == pygame.K_s:
                    env.env_history.save_video("videos", "test", fps=2)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()


if __name__ == "__main__":
    test()

    # pygame.time.delay(2000)

