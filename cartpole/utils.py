import numpy as np
from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.sparse.linalg import cg
from jax.lax import stop_gradient

import haiku as hk
from absl import flags
FLAGS = flags.FLAGS


def evaluate(agent, eval_env, rng, num_eval_episodes=10):
  average_episode_reward = 0
  for episode in range(num_eval_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
      rng, _ = jax.random.split(rng)
      action = agent.act(agent.params_Q, obs, rng).item()
      obs, reward, done, _ = eval_env.step(action)
      episode_reward += reward
    average_episode_reward += episode_reward
  average_episode_reward /= num_eval_episodes
  return average_episode_reward

@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def root_solve(param_func, init_xs, params, replay, rng, solvers):
  # to mimic two_phase_solve API
  fwd_solver = solvers[0]
  return fwd_solver(param_func, init_xs, params, replay, rng)

def root_solve_fwd(param_func, init_xs, params, replay, rng, solvers):
  sol = root_solve(param_func, init_xs, params, replay, rng, solvers)
  tpQ = jax.lax.stop_gradient(sol.target_params_Q)
  return sol, (sol.params_Q, params, replay, rng, tpQ)

def root_solve_bwd(param_func, solvers, res, g):
  pQ, params, replay, rng, tpQ = res
  _, vdp_fun = jax.vjp(lambda y: param_func(y, pQ, replay, rng, tpQ), params)
  g_main = g[0] if isinstance(g, tuple) else g
  if FLAGS.with_inv_jac:
    # _, vds_fun = jax.vjp(lambda x: param_func(params, x), pQ)
    # (J)^-1 -> (J+cI)^-1
    _, vds_fun = jax.vjp(lambda x: jax.tree_multimap(
      lambda y,z: y + 1e-5*z, param_func(params, x, replay, rng, tpQ), x), pQ)
    vdsinv = cg(lambda z: vds_fun(z)[0], g_main, maxiter=100)[0]
    vdp = vdp_fun(vdsinv)[0]
  else:
    vdp = vdp_fun(g_main)[0]
  z_sol, z_replay, z_rng = jax.tree_map(jnp.zeros_like, (pQ, replay, rng))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng

root_solve.defvjp(root_solve_fwd, root_solve_bwd)

def add_dict(d, k, v):
  if not isinstance(v, list):
    v = [v]
  if k in d:
    d[k].extend(v)
  else:
    d[k] = v

@jax.jit
def soft_update_params(tau, params, target_params):
  return jax.tree_multimap(
    lambda p, tp: tau * p + (1 - tau) * tp, 
    params, target_params)

@jax.jit
def tree_norm(tree):
  return jnp.sqrt(sum((x**2).sum() for x in jax.tree_leaves(tree)))

def net_fn(net_type, dims, x):
  obs_dim, action_dim, hidden_dim = dims
  activation = jax.nn.relu
  init = hk.initializers.Orthogonal(scale=jnp.sqrt(2.0))
  layers = [
    hk.Linear(hidden_dim, w_init=init), activation,
    hk.Linear(hidden_dim, w_init=init), activation,
  ]
  final_init = hk.initializers.Orthogonal(scale=1e-2)
  # T -- model, Q and V -- value functions
  if FLAGS.agent_type == 'vep':
    if net_type == 'V':
      ensemble = []
      for i in range(FLAGS.num_ensemble_vep):
        layers = [
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(1, w_init=final_init)
        ]
        mlp = hk.Sequential(layers)
        ensemble.append(mlp(x))
      return ensemble
  if net_type == 'V':
    layers += [hk.Linear(1, w_init=final_init)] 
  elif net_type == 'Q':
    layers += [hk.Linear(action_dim, w_init=final_init)]
  elif net_type == 'T':
    out_dim = 2 * obs_dim if FLAGS.prob_model else obs_dim
    layers += [hk.Linear(out_dim, w_init=final_init)]

  if net_type == 'Q' and not FLAGS.no_double:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(action_dim, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  elif net_type == 'T' and not FLAGS.no_learn_reward:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(1, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  else:
    mlp = hk.Sequential(layers)
    return mlp(x)

def init_net_opt(net_type, dims):
  net = hk.without_apply_rng(hk.transform(partial(net_fn, net_type, dims)))
  if net_type == 'Q':
    opt = optax.adam(FLAGS.inner_lr)
  else:
    opt = optax.adam(FLAGS.lr)
  Model = namedtuple(net_type, 'net opt')
  return Model(net, opt)
