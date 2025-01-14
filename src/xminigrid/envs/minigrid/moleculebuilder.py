import pickle

import jax
import jax.numpy as jnp
import timeit
from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentHoldGoal
from ...core.grid import (
    empty_world,
    sample_direction,
)
from ...core.rules import TileNearRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State
from tqdm import tqdm


class MoleculeBuilder(Environment):
    def __init__(self):
        super().__init__()
        start_time = timeit.default_timer()
        with open("pkls/id_to_combination.pkl", "rb") as f:
            id_to_combination = pickle.load(f)
        # with open("envs/env_easy.pkl", "rb") as f:
        #     self.dic = pickle.load(f)
        self.dic = {k: self.dic[k] for k in list(self.dic)[:128]}
        self.rules_lst = []
        for k, lst in self.dic.items():
            for d in lst:
                if "rules" in d:  # TODO: limit rules benchmark
                    inst_rules = []
                    for rule, res in d["rules"].items():
                        a_1, a_2 = rule
                        tile_a = id_to_combination[a_1]
                        tile_b = id_to_combination[a_2]
                        prod_tile = id_to_combination[res]
                        rule = TileNearRule(
                            tile_a=tile_a, tile_b=tile_b, prod_tile=prod_tile
                        )
                        inst_rules.append(rule.encode())
                        if len(inst_rules) == 9:
                            break
                    self.rules_lst.append(jnp.stack(inst_rules))

        self.rule_encoding = jnp.array([inst_rules for inst_rules in self.rules_lst])
        self.arr_agent = []
        self.arr_grid = []
        self.arr_goals = []
        for k, v_list in self.dic.items():
            for v in v_list:
                if "size" in v:
                    x = v["size"][0]
                    y = v["size"][1]
                    g = empty_world(x + 1, y)
                    for walls_xy in v["walls"]:
                        g = g.at[walls_xy[0], walls_xy[1]].set(
                            TILES_REGISTRY[Tiles.WALL, Colors.WHITE]
                        )
                    for xy, atom_id in v["atoms"].items():
                        g = g.at[xy[0], xy[1]].set(
                            TILES_REGISTRY[id_to_combination[atom_id]]
                        )
                    for xy, obstacle_id in v["obstacles"].items():
                        g = g.at[xy[0], xy[1]].set(
                            TILES_REGISTRY[id_to_combination[obstacle_id]]
                        )
                    ((tup, final_mol),) = v["win_condition"].items()
                    final_1, final_2 = tup
                    final_1_tile = TILES_REGISTRY[id_to_combination[final_1]]
                    final_2_tile = TILES_REGISTRY[id_to_combination[final_2]]
                    final_mol = TILES_REGISTRY[id_to_combination[final_mol]]
                    g = g.at[x, 0].set(final_1_tile)
                    g = g.at[x, y - 1].set(final_2_tile)
                    self.arr_goals.append(AgentHoldGoal(tile=final_mol).encode())
                    self.arr_grid.append(g)
                    self.arr_agent.append(v["agent"])

        self.arr_agent = jnp.array(self.arr_agent)
        self.arr_grid = jnp.stack(self.arr_grid)
        self.arr_goals = jnp.array(self.arr_goals)

        elapsed = timeit.default_timer() - start_time
        print(f"init done {elapsed=}")

    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=6, width=11)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 1000

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        key, *keys = jax.random.split(key, num=7)
        # agent_coords, grid = random.choice(self.agent_to_grid)
        # print(agent_coords)
        index = jax.random.choice(keys[2], jnp.arange(self.arr_agent.shape[0]))
        agent_coords = self.arr_agent[index]
        grid = self.arr_grid[index]
        rules = self.rule_encoding[index]
        # pickup_obj, block_obj = jax.random.choice(keys[0], _allowed_entities, shape=(2,))
        # obj = jax.random.choice(keys[0], _allowed_entities)
        # door_color, obj_color = jax.random.choice(keys[1], _allowed_colors, shape=(2,))
        # jax.debug.print("x: {}", key)
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
        goal_encoding = self.arr_goals[index]
        # goal_encoding_old = AgentHoldGoal(tile=TILES_REGISTRY[pickup_obj, block_obj]).encode()
        # jax.debug.print("x: {}", goal_encoding)
        # jax.debug.print("x: {}", goal_encoding_old)
        print(rules.shape)
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=goal_encoding,
            rule_encoding=rules,
            carry=EnvCarry(),
        )
        return state


class MoleculeBuilderEasy(MoleculeBuilder):
    def __init__(self):
        with open("envs/env_easy.pkl", "rb") as f:
            self.dic = pickle.load(f)
        super().__init__()


class MoleculeBuilderHard(MoleculeBuilder):
    def __init__(self):
        with open("envs/env_hard.pkl", "rb") as f:
            self.dic = pickle.load(f)
        super().__init__()
