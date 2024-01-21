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


# alternatively users can provide step_fn and reset_fn instead
# of the closure, but in whis way it is simpler to use after the creation
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
        rng, transitions = jax.lax.scan(
            _step_fn, (rng, timestep), None, length=num_steps
        )
        return transitions

    return rollout


env, env_params = xminigrid.make("MiniGrid-EmptyRandom-8x8")
# do not forget to use auto reset wrapper!
env = GymAutoResetWrapper(env)

# jiting the entire rollout
rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=1000))

# first execution will compile
transitions = rollout_fn(jax.random.PRNGKey(0))

# vmap_rollout = jax.jit(jax.vmap(build_rollout(env, env_params, num_steps=1000)))
# rngs = jax.random.split(jax.random.PRNGKey(0), num=1024)
# vmap_transitions = vmap_rollout(rngs)

print(f"Transitions shapes: \n, {jtu.tree_map(jnp.shape, transitions)}")

# optionally render the state
images = []

for i in trange(1000):
    timestep = jtu.tree_map(lambda x: x[i], transitions)
    images.append(env.render(env_params, timestep))

imageio.mimsave("example_rollout.mp4", images, fps=32, format="mp4")
