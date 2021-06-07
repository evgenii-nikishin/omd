from typing import Tuple

import jax.numpy as jnp

from flax import linen as nn
from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp
from jax_rl.networks.common import InfoDict


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(omd: ModelActorCriticTemp, entropy: float,
           target_entropy: float) -> Tuple[ModelActorCriticTemp, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = omd.temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = omd.temp.apply_gradient(temperature_loss_fn)

    new_sac = omd.replace(temp=new_temp)

    return new_sac, info
