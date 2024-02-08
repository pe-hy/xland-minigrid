# relative import of ../src/xminigrid
import sys
#sys.path.append("../")

#from src import xminigrid

import xminigrid
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import timeit
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

from xminigrid.wrappers import GymAutoResetWrapper

# run this and see example_rollout.mp4

def build_rollout(env, env_params, num_steps):
    def rollout(rng):
        def _step_fn(carry, _):
            rng, timestep = carry
            rng, _rng = jax.random.split(rng)
            action = jax.random.randint(
                _rng, shape=(), minval=0, maxval=env.num_actions(env_params)
            )

            timestep = env.step(env_params, timestep, action)
            return (rng, timestep), timestep

        rng, _rng = jax.random.split(rng)

        timestep = env.reset(env_params, _rng)
        #jax.debug.print('x:{}',rng)

        rng, transitions = jax.lax.scan(
            _step_fn, (rng, timestep), None, length=num_steps
        )
        return transitions

    return rollout

def print_size(x):
    print(len(x))

env, env_params = xminigrid.make("MiniGrid-UnlockPickUp") # env_params will be width and heigth
# do not forget to use auto reset wrapper!
env = GymAutoResetWrapper(env)

# jiting the entire rollout
#rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=1000))

# first execution will compile
#transitions = rollout_fn(jax.random.PRNGKey(0))
#timestep = jtu.tree_map(print_size, transitions)
num_steps = 1000
vmap_rollout = jax.jit(jax.vmap(build_rollout(env, env_params, num_steps=num_steps)))
rngs = jax.random.split(jax.random.PRNGKey(0), num=2)
vmap_transitions = vmap_rollout(rngs)

#timestep = jtu.tree_map(lambda x: x[0][1], vmap_transitions)

#print(f"Transitions shapes: \n, {jtu.tree_map(jnp.shape, vmap_transitions)}")

# optionally render the state
images = []
instance_to_render = 0

for i in trange(num_steps):
    timestep = jtu.tree_map(lambda x: x[instance_to_render][i], vmap_transitions)
    images.append(env.render(env_params, timestep))
imageio.mimsave(f"example_rollout{instance_to_render}.mp4", images, fps=32, format="mp4")


images = []
instance_to_render = 1

for i in trange(num_steps):
    timestep = jtu.tree_map(lambda x: x[instance_to_render][i], vmap_transitions)
    images.append(env.render(env_params, timestep))
imageio.mimsave(f"example_rollout{instance_to_render}.mp4", images, fps=32, format="mp4")