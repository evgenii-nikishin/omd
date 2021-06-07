from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params


def update(omd: ModelActorCriticTemp,
           batch: Batch) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(omd.rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = omd.actor.apply({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1, q2 = omd.critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * omd.temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = omd.actor.apply_gradient(actor_loss_fn)

    new_omd = omd.replace(actor=new_actor, rng=rng)

    return new_omd, info
