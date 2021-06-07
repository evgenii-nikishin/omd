from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import distrax
import flax.linen as nn
from jax_rl.networks.common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init())(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), 1))


@jax.partial(jax.jit, static_argnums=1)
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
