# relative import of ../src/xminigrid
import sys
import timeit

import imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np

# sys.path.append("../")
# from src import xminigrid
import xminigrid
from tqdm.auto import tqdm, trange
from xminigrid.wrappers import GymAutoResetWrapper

# run this and see example_rollout.mp4


def build_rollout(env, env_params, num_steps):
    def rollout(rng):
        def _step_fn(carry, _):
            rng, timestep = carry
            rng, _rng = jax.random.split(rng)
            action = jax.random.randint(_rng, shape=(), minval=0, maxval=env.num_actions(env_params))

            timestep = env.step(env_params, timestep, action)
            return (rng, timestep), timestep

        rng, _rng = jax.random.split(rng)

        timestep = env.reset(env_params, _rng)
        # jax.debug.print('x:{}',rng)

        rng, transitions = jax.lax.scan(_step_fn, (rng, timestep), None, length=num_steps)
        return transitions

    return rollout


env, env_params = xminigrid.make("MiniGrid-MoleculeBuilder")  
# do not forget to use auto reset wrapper!
env = GymAutoResetWrapper(env)

# jiting the entire rollout
# rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=1000))

# first execution will compile
# transitions = rollout_fn(jax.random.PRNGKey(0))
# timestep = jtu.tree_map(print_size, transitions)
num_steps = 1000
print("vmap")
vmap_rollout = jax.jit(jax.vmap(build_rollout(env, env_params, num_steps=num_steps)))
rngs = jax.random.split(jax.random.PRNGKey(0), num=2)
vmap_transitions = vmap_rollout(rngs)
print("vmap_end")
# timestep = jtu.tree_map(lambda x: x[0][1], vmap_transitions)

# print(f"Transitions shapes: \n, {jtu.tree_map(jnp.shape, vmap_transitions)}")

# optionally render the state
images = []
instance_to_render = 0
for i in trange(num_steps):
    timestep = jtu.tree_map(lambda x: x[instance_to_render][i], vmap_transitions)
    images.append(env.render(env_params, timestep))
    
    if i == 1000:
        break

imageio.mimsave(f"example_rollout{instance_to_render}.mp4", images, fps=15, format="mp4")



images = []
instance_to_render = 1
print("herex2")
for i in trange(num_steps):
    timestep = jtu.tree_map(lambda x: x[instance_to_render][i], vmap_transitions)
    images.append(env.render(env_params, timestep))
imageio.mimsave(f"example_rollout{instance_to_render}.mp4", images, fps=100, format="mp4")
