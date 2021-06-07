from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params


def target_update(omd: ModelActorCriticTemp,
                  tau: float) -> ModelActorCriticTemp:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), omd.critic.params,
        omd.target_critic.params)

    new_target_critic = omd.target_critic.replace(params=new_target_params)

    return omd.replace(target_critic=new_target_critic)


def update(omd: ModelActorCriticTemp,
           model_params: Params,
           batch: Batch,
           discount: float,
           soft_critic: bool,
           use_model: bool,
           return_grad: bool = False) -> Tuple[ModelActorCriticTemp, InfoDict]:
    if use_model:
        next_observations, rewards = omd.model.apply({'params': model_params},
                                                     batch.observations,
                                                     batch.actions)
    else:
        next_observations, rewards = batch.next_observations, batch.rewards

    dist = omd.actor(jax.lax.stop_gradient(next_observations))
    rng, key = jax.random.split(omd.rng)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q1, next_q2 = omd.target_critic(next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = rewards + discount * batch.masks * next_q

    if soft_critic:
        target_q -= discount * batch.masks * omd.temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = omd.critic.apply({'params': critic_params},
                                  batch.observations, batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    if return_grad:
        grads, _ = jax.grad(critic_loss_fn, has_aux=True)(omd.critic.params)
        return grads

    new_critic, info = omd.critic.apply_gradient(critic_loss_fn)

    new_omd = omd.replace(critic=new_critic, rng=rng)

    if use_model:
        return new_omd, {**info, 'r_pred': rewards.mean()}
    else:
        return new_omd, info
