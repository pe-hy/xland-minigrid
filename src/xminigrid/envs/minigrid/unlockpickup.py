import os
import pickle
import random

import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentHoldGoal
from ...core.grid import coordinates_mask, empty_world, sample_coordinates, sample_direction, two_rooms
from ...core.rules import EmptyRule, TileNearRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# colors like in the original minigrid
_allowed_colors = jnp.array(
    (
        Colors.CYAN,
        # Colors.MAGENTA,
        # Colors.LIME,
        # Colors.TEAL,
        # Colors.MAROON,
    )
)
_allowed_entities = jnp.array(
    (
        # Tiles.DIAMOND,
        # Tiles.PARALLELOGRAM,
        # Tiles.PENTAGON,
        Tiles.CROSS,
        # Tiles.TRAPEZOID,
    )
)
_rule_encoding = EmptyRule().encode()[None, ...]


class UnlockPickUp(Environment):
    def __init__(self):
        super().__init__()

        with open("pkls/id_to_combination.pkl", "rb") as f:
            id_to_combination = pickle.load(f)
        self.dic = {
            0: [
                {
                    "size": (10, 10),
                    "walls": [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (0, 7),
                        (0, 8),
                        (0, 9),
                        (9, 0),
                        (8, 0),
                        (7, 0),
                        (6, 0),
                        (5, 0),
                        (4, 0),
                        (3, 0),
                        (2, 0),
                        (1, 0),
                        (9, 1),
                        (9, 2),
                        (9, 3),
                        (9, 4),
                        (9, 5),
                        (9, 6),
                        (9, 7),
                        (9, 8),
                        (9, 9),
                        (1, 9),
                        (2, 9),
                        (3, 9),
                        (4, 9),
                        (5, 9),
                        (6, 9),
                        (7, 9),
                        (8, 9),
                    ],
                    "atoms": {10: (4, 5), 11: (6, 6), 12: (8, 5), 13: (7, 2)},
                    "obstacles": {20: (1, 5)},
                    "agent": (2, 2),
                },
                {"rules": {(10, 11): 10002, (12, 13): 10001}},
            ],
            1: [
                {
                    "size": (10, 10),
                    "walls": [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (0, 7),
                        (0, 8),
                        (0, 9),
                        (9, 0),
                        (8, 0),
                        (7, 0),
                        (6, 0),
                        (5, 0),
                        (4, 0),
                        (3, 0),
                        (2, 0),
                        (1, 0),
                        (9, 1),
                        (9, 2),
                        (9, 3),
                        (9, 4),
                        (9, 5),
                        (9, 6),
                        (9, 7),
                        (9, 8),
                        (9, 9),
                        (1, 9),
                        (2, 9),
                        (3, 9),
                        (4, 9),
                        (5, 9),
                        (6, 9),
                        (7, 9),
                        (8, 9),
                    ],
                    "atoms": {10: (3, 3), 11: (4, 4), 12: (5, 5), 13: (6, 6)},
                    "obstacles": {20: (7, 7)},
                    "agent": (8, 8),
                },
                {"rules": {(10, 11): 10001, (12, 13): 10002}},
            ],
        }
        rules_lst = []
        for k, lst in self.dic.items():
            for d in lst:
                if "rules" in d:
                    for rule, res in d["rules"].items():
                        a_1, a_2 = rule
                        tile_a = id_to_combination[a_1]
                        tile_b = id_to_combination[a_2]
                        prod_tile = id_to_combination[res]
                        rule = TileNearRule(tile_a=tile_a, tile_b=tile_b, prod_tile=prod_tile)
                        rules_lst.append(rule)

        for rule in rules_lst:
            rule.encode()

        self.arr_agent = []
        self.arr_grid = []

        for k, v_list in self.dic.items():
            for v in v_list:
                if "size" in v:
                    x = v["size"][0]
                    y = v["size"][1]
                    g = empty_world(x, y)
                    for walls_xy in v["walls"]:
                        g = g.at[walls_xy[0], walls_xy[1]].set(TILES_REGISTRY[Tiles.WALL, Colors.WHITE])
                    for atom_id, xy in v["atoms"].items():
                        g = g.at[xy[0], xy[1]].set(TILES_REGISTRY[id_to_combination[atom_id]])
                    for obstacle_id, xy in v["obstacles"].items():
                        g = g.at[xy[0], xy[1]].set(TILES_REGISTRY[id_to_combination[obstacle_id]])
                    self.arr_agent.append(v["agent"])
                    self.arr_grid.append(g)
        self.arr_agent = jnp.array(self.arr_agent)
        self.arr_grid = jnp.stack(self.arr_grid)

    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=6, width=11)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 8 * params.height**2

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        print(type(_rule_encoding))
        key, *keys = jax.random.split(key, num=7)

        # agent_coords, grid = random.choice(self.agent_to_grid)
        # print(agent_coords)
        index = jax.random.choice(keys[2], jnp.arange(self.arr_agent.shape[0]))
        agent_coords = self.arr_agent[index]
        grid = self.arr_grid[index]
        # jax.debug.print("x: {}", key)
        obj = jax.random.choice(keys[0], _allowed_entities)
        door_color, obj_color = jax.random.choice(keys[1], _allowed_colors, shape=(2,))
        # door_pos = jax.random.randint(keys[2], shape=(), minval=1, maxval=params.height - 1)

        # grid = two_rooms(params.height, params.width)
        # grid = grid.at[door_pos, params.width // 2].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, door_color])

        # mask out positions after the wall, so that agent and key are always on the same side
        # WARN: this is a bit expensive, judging by the FPS benchmark
        # mask = coordinates_mask(grid, (params.height, params.width // 2), comparison_fn=jnp.less)
        # mask = coordinates_mask(grid, (0, params.width // 2 + 1), comparison_fn=jnp.greater_equal)
        # obj_coords = sample_coordinates(keys[4], grid, num=1, mask=mask).squeeze()
        # grid = grid.at[key_coords[0], key_coords[1]].set(TILES_REGISTRY[Tiles.KEY, door_color])

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[5]))
        goal_encoding = AgentHoldGoal(tile=TILES_REGISTRY[obj, obj_color]).encode()

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
