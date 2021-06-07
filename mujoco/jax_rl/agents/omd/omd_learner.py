"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags

import optax
from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp
from jax_rl.agents.omd import actor, critic, model, temperature
from jax_rl.datasets import Batch
from jax_rl.networks import critic_net, models, policies
from jax_rl.networks.common import InfoDict, TrainState

FLAGS = flags.FLAGS


@jax.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _update_jit_mle(mle: ModelActorCriticTemp, batch: Batch, discount: float,
                    tau: float, target_entropy: float, update_target: bool,
                    inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:

    mle, model_info = model.update_mle(mle, batch, discount, tau, inner_steps)

    for _ in range(FLAGS.config.inner_steps):
        mle, critic_info = critic.update(mle,
                                         mle.model.params,
                                         batch,
                                         discount,
                                         soft_critic=True,
                                         use_model=True)

        if update_target:
            mle = critic.target_update(mle, tau)

        # actor does not use next_observations and rewards
        mle, actor_info = actor.update(mle, batch)
        mle, alpha_info = temperature.update(mle, actor_info['entropy'],
                                             target_entropy)

    return mle, {**model_info, **critic_info, **actor_info, **alpha_info}


@jax.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _update_jit_omd(omd: ModelActorCriticTemp, batch: Batch, discount: float,
                    tau: float, target_entropy: float, update_target: bool,
                    inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:

    # critic, actor, and temp are updated within the model update
    omd, merged_info = model.update_omd(omd, batch, discount, tau,
                                        target_entropy, inner_steps)

    return omd, {**merged_info}


class OMDLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 model_lr: float = 3e-4,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dim: int = 256,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 inner_steps: int = 1,
                 algo: str = 'omd'):

        action_dim = actions.shape[-1]
        obs_dim = observations.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.inner_steps = inner_steps
        self.algo = algo

        make_tx = lambda lr: optax.adam(learning_rate=lr)

        rng = jax.random.PRNGKey(seed)
        rng, model_key, actor_key, critic_key, temp_key = jax.random.split(
            rng, 5)

        model_hidden_dims = tuple(model_hidden_dim for _ in hidden_dims)
        model_def = models.DetModel(model_hidden_dims, obs_dim)
        model = TrainState.create(model_def,
                                  inputs=[model_key, observations, actions],
                                  tx=make_tx(model_lr))

        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = TrainState.create(actor_def,
                                  inputs=[actor_key, observations],
                                  tx=make_tx(actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = TrainState.create(critic_def,
                                   inputs=[critic_key, observations, actions],
                                   tx=make_tx(critic_lr))
        target_critic = TrainState.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = TrainState.create(temperature.Temperature(init_temperature),
                                 inputs=[temp_key],
                                 tx=make_tx(temp_lr))

        self.omd = ModelActorCriticTemp(model=model,
                                        actor=actor,
                                        critic=critic,
                                        target_critic=target_critic,
                                        temp=temp,
                                        rng=rng)
        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.omd.rng,
                                               self.omd.actor.apply_fn,
                                               self.omd.actor.params,
                                               observations, temperature)

        self.omd = self.omd.replace(rng=rng)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.algo == 'omd':
            upd_func = _update_jit_omd
        elif self.algo == 'mle':
            upd_func = _update_jit_mle

        self.step += 1
        self.omd, info = upd_func(self.omd, batch, self.discount, self.tau,
                                  self.target_entropy,
                                  self.step % self.target_update_period == 0,
                                  self.inner_steps)
        return info
