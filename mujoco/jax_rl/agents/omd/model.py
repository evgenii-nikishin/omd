from collections.abc import Callable
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from absl import flags

from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp
from jax_rl.agents.omd import actor, critic, temperature
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params, TrainState

FLAGS = flags.FLAGS


@partial(jax.custom_vjp, nondiff_argnums=(0, 7))
def root_solve(param_func: Callable, init_xs: ModelActorCriticTemp,
               params: Params, batch: Batch, discount: float, tau: float,
               target_entropy: float, solvers: Tuple[Callable, ]):

    fwd_solver = solvers[0]
    return fwd_solver(param_func, init_xs, params, batch, discount, tau,
                      target_entropy)


def root_solve_fwd(param_func: Callable, init_xs: ModelActorCriticTemp,
                   params: Params, batch: Batch, discount: float, tau: float,
                   target_entropy: float, solvers: Tuple[Callable, ]):
    sol = root_solve(param_func, init_xs, params, batch, discount, tau,
                     target_entropy, solvers)
    new_omd, merged_info = sol
    return sol, (new_omd, params, batch, discount, tau, target_entropy)


def root_solve_bwd(param_func: Callable, solvers: Tuple[Callable, ],
                   res: Tuple[Params, ], g: Tuple[ModelActorCriticTemp,
                                                  InfoDict]):
    # only the identity approximation version
    new_omd, params, batch, discount, tau, target_entropy = res

    _, vdp_fun = jax.vjp(
        lambda y: param_func(new_omd, y, batch, discount, tau, target_entropy),
        params)
    # g contains the adjoint for output of critic.update
    g_main = g[0].critic.params
    vdp = vdp_fun(g_main)[0]

    z_new_omd, z_batch, z_discount, z_tau, z_tent = jax.tree_map(
        jnp.zeros_like, (new_omd, batch, discount, tau, target_entropy))
    return z_new_omd, jax.tree_map(lambda x: -x,
                                   vdp), z_batch, z_discount, z_tau, z_tent


root_solve.defvjp(root_solve_fwd, root_solve_bwd)


def constraint_func(omd: ModelActorCriticTemp, model_params: Params,
                    batch: Batch, discount: float, tau: float,
                    target_entropy: float):
    """Get grad_Q (model-Bellman-error) = 0 constraint.
    """

    return critic.update(omd,
                         model_params,
                         batch,
                         discount,
                         soft_critic=True,
                         use_model=True,
                         return_grad=True)


def fwd_solver(constraint_func: Callable, omd: ModelActorCriticTemp,
               model_params: Params, batch: Batch, discount: float, tau: float,
               target_entropy: float):
    """Get Q_* satisfying the constraint (approximately). Makes K grad updates.
    """

    for _ in range(FLAGS.config.inner_steps):
        omd, critic_info = critic.update(omd,
                                         model_params,
                                         batch,
                                         discount,
                                         soft_critic=True,
                                         use_model=True)
        omd = critic.target_update(omd, tau)

        # note that actor and temp do not use next_observations and rewards
        omd, actor_info = actor.update(omd, batch)
        omd, alpha_info = temperature.update(omd, actor_info['entropy'],
                                             target_entropy)

    merged_info = {**critic_info, **actor_info, **alpha_info}
    return omd, merged_info


def update_omd(omd: ModelActorCriticTemp, batch: Batch, discount: float,
               tau: float, target_entropy: float,
               inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(omd.rng)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Update critic_params on a batch w.r.t. loss yielded by model_params
        new_omd, merged_info = root_solve(constraint_func, omd, model_params,
                                          batch, discount, tau, target_entropy,
                                          (fwd_solver, ))

        _, model_info = critic.update(new_omd,
                                      None,
                                      batch,
                                      discount,
                                      soft_critic=True,
                                      use_model=False)

        return model_info['critic_loss'], ({
            **merged_info, 'model_loss':
            model_info['critic_loss']
        }, new_omd)

    new_model, (info, new_omd) = omd.model.apply_gradient(model_loss_fn)
    new_omd = new_omd.replace(model=new_model, rng=rng)

    return new_omd, info


def update_mle(mle: ModelActorCriticTemp, batch: Batch, discount: float,
               tau: float,
               inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(mle.rng)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        next_observations, rewards = mle.model.apply({'params': model_params},
                                                     batch.observations,
                                                     batch.actions)

        model_loss = ((next_observations -
                       batch.next_observations)**2).sum(-1).mean()
        model_loss += ((rewards - batch.rewards)**2).mean()

        return model_loss, {'model_loss': model_loss}

    new_model, info = mle.model.apply_gradient(model_loss_fn)

    mle = mle.replace(model=new_model, rng=rng)

    return mle, info
