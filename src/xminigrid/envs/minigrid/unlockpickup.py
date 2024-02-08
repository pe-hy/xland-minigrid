import os
import pickle

import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentHoldGoal
from ...core.grid import coordinates_mask, empty_world, sample_coordinates, sample_direction, two_rooms
from ...core.rules import EmptyRule
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
                    "mols": [100, 101],
                    "obstacles": {20: (1, 5)},
                    "agent": (2, 2),
                },
                {"rules": {(10, 11): 100, (12, 13): 101}, "win_condition": (100, 101)},
            ]
        }
        for k, v_list in self.dic.items():
            for v in v_list:
                if "size" in v:
                    x = v["size"][0]
                    y = v["size"][1]
                    g = empty_world(x, y)
                    for walls_xy in v["walls"]:
                        g = g.at[walls_xy[0], walls_xy[1]].set(TILES_REGISTRY[Tiles.WALL, Colors.WHITE])
                    for atom_id, xy in v["atoms"].items():
                        g = g.at[xy[0], xy[1]].set(
                            TILES_REGISTRY[id_to_combination[atom_id][0], id_to_combination[atom_id][1]]
                        )
        self.grids = jnp.array([g])

    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=6, width=11)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 8 * params.height**2

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        key, *keys = jax.random.split(key, num=7)
        grid = jax.random.choice(keys[2], self.grids)
        # jax.debug.print("x: {}", key)
        obj = jax.random.choice(keys[0], _allowed_entities)
        door_color, obj_color = jax.random.choice(keys[1], _allowed_colors, shape=(2,))
        # door_pos = jax.random.randint(keys[2], shape=(), minval=1, maxval=params.height - 1)

        # grid = two_rooms(params.height, params.width)
        # grid = grid.at[door_pos, params.width // 2].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, door_color])

        # mask out positions after the wall, so that agent and key are always on the same side
        # WARN: this is a bit expensive, judging by the FPS benchmark
        mask = coordinates_mask(grid, (params.height, params.width // 2), comparison_fn=jnp.less)

        # mask = coordinates_mask(grid, (0, params.width // 2 + 1), comparison_fn=jnp.greater_equal)
        obj_coords = sample_coordinates(keys[4], grid, num=1, mask=mask).squeeze()

        # grid = grid.at[key_coords[0], key_coords[1]].set(TILES_REGISTRY[Tiles.KEY, door_color])
        grid = grid.at[obj_coords[0], obj_coords[1]].set(TILES_REGISTRY[obj, obj_color])
        agent_coords = self.agent_coords_dict[grid]
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
