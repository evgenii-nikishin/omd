import numpy as np
from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy import stats
import optax
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient
from jax.experimental.host_callback import id_tap

from utils import *
from replay_buffer import ReplayBuffer
from absl import flags
import chex
FLAGS = flags.FLAGS

AuxP = namedtuple('AuxP', 'params_Q target_params_Q opt_state_Q rng')
AuxOut = namedtuple('AuxOut', 'vals_Q entropy_Q next_obs_nll')

nll_loss = lambda x, m, ls: -stats.norm.logpdf(x, m, jnp.exp(ls)).sum(-1).mean()
mse_loss = lambda x, xp: ((x - xp) ** 2).sum(-1).mean()


class Agent:
  def __init__(self, obs_space, action_space):
    self.obs_dim = obs_space.shape[0]
    self.action_dim = action_space.n
    self.obs_range = (obs_space.low, obs_space.high)
    demo_obs = jnp.ones((1, self.obs_dim))
    demo_obs_action = jnp.ones((1, self.obs_dim + self.action_dim))
    self.rngs = hk.PRNGSequence(FLAGS.seed)
    
    if FLAGS.agent_type == 'vep':
      self.V = init_net_opt('V', (self.obs_dim, self.action_dim, FLAGS.hidden_dim))
      self.params_V = self.V.net.init(next(self.rngs), demo_obs)

    self.T = init_net_opt('T', (self.obs_dim, self.action_dim, FLAGS.model_hidden_dim))
    self.params_T = self.T.net.init(next(self.rngs), demo_obs_action)
    self.opt_state_T = self.T.opt.init(self.params_T)

    self.Q = init_net_opt('Q', (self.obs_dim, self.action_dim, FLAGS.hidden_dim))
    self.params_Q = self.target_params_Q = self.Q.net.init(next(self.rngs), demo_obs)
    self.opt_state_Q = self.Q.opt.init(self.params_Q)

  @partial(jax.jit, static_argnums=(0,))
  def act(self, params_Q, obs, rng):
    obs = jnp.array(obs) if isinstance(obs, list) else obs[None, ...] 
    current_Q = self.Q.net.apply(params_Q, obs[None, ...])
    if not FLAGS.no_double:
      current_Q = 0.5 * (current_Q[0] + current_Q[1])
    if FLAGS.hard:
      action = jnp.argmax(current_Q, axis=-1)
    else:
      action = jax.random.categorical(rng, current_Q / FLAGS.alpha)
    return action if isinstance(obs, list) else action[0]

  @partial(jax.jit, static_argnums=(0,))
  def model_pred(self, params_T, obs, action, rng):
    if not isinstance(action, int):
      action = action[:, 0]
    a = jax.nn.one_hot(action, self.action_dim)
    x = jnp.concatenate((obs, a), axis=-1)
    if FLAGS.prob_model:
      if FLAGS.no_learn_reward:
        next_obs_pred = self.T.net.apply(params_T, x)
        reward_pred = None
      else:
        next_obs_pred, reward_pred = self.T.net.apply(params_T, x)
      means, logstds = next_obs_pred.split(2, axis=1)
      logstds = jnp.clip(logstds, -5.0, 2.0)
      noise = jax.random.normal(rng, shape=means.shape)
      samples = noise * jnp.exp(logstds) + means
      return samples, means, logstds, reward_pred
    else:
      if FLAGS.no_learn_reward:
        next_obs_pred = self.T.net.apply(params_T, x)
        reward_pred = None
      else:
        next_obs_pred, reward_pred = self.T.net.apply(params_T, x)
      return next_obs_pred, next_obs_pred, None, reward_pred
    
  def batch_real_to_model(self, params_T, batch, rng):
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    next_obs_pred, means, logstds, reward_pred = self.model_pred(
        params_T, obs, action, rng)
    if FLAGS.no_learn_reward:
      reward_pred = reward
    batch_model = obs, action, reward_pred, next_obs_pred, not_done, not_done_no_max
    if FLAGS.prob_model:
      nll = nll_loss(next_obs, means, logstds)
    else:
      nll = mse_loss(next_obs_pred, next_obs)
    return batch_model, nll
  
  @partial(jax.jit, static_argnums=(0,))
  @chex.assert_max_traces(n=1)
  def loss_Q(self, params_Q, target_params_Q, batch):
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    
    target_Q = self.Q.net.apply(stop_gradient(target_params_Q), next_obs)
    if FLAGS.hard:
      if FLAGS.no_double:
        target_V = jnp.max(target_Q, axis=-1, keepdims=True)
      else:
        target_Q = jnp.minimum(target_Q[0], target_Q[1])
        target_V = jnp.max(target_Q, axis=-1, keepdims=True)
    else:
      if FLAGS.no_double:
        target_V = FLAGS.alpha * logsumexp(target_Q / FLAGS.alpha, 
                                          axis=-1, keepdims=True)
      else:
        target_Q = jnp.minimum(target_Q[0], target_Q[1])
        target_V = FLAGS.alpha * logsumexp(target_Q / FLAGS.alpha, 
                                          axis=-1, keepdims=True)
    
    target_Q = (reward + (not_done_no_max * FLAGS.discount * target_V))[:, 0]
    
    Q_s = self.Q.net.apply(params_Q, obs)
    if FLAGS.no_double:
      current_Q = Q_s[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      vals_Q = current_Q.mean()
      entropy_Q = (-jax.nn.log_softmax(Q_s) * jax.nn.softmax(Q_s)).sum(-1).mean()
      mse_Q = jnp.mean((current_Q - target_Q)**2)
    else:
      current_Q1 = Q_s[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      current_Q2 = Q_s[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      entropy_Q  = (-jax.nn.log_softmax(Q_s[0]) * jax.nn.softmax(Q_s[0])).sum(-1).mean()
      entropy_Q += (-jax.nn.log_softmax(Q_s[1]) * jax.nn.softmax(Q_s[1])).sum(-1).mean()
      vals_Q = 0.5 * (current_Q1.mean() + current_Q2.mean())
      mse_Q = 0.5 * (jnp.mean((current_Q1 - target_Q)**2) + jnp.mean((current_Q2 - target_Q)**2))

    aux_out = AuxOut(vals_Q, entropy_Q, None)
    return mse_Q, aux_out

  def loss_mle(self, params_T, batch, rng):
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
    assert next_obs.ndim == pred.ndim  # no undesired broadcasting
    
    nll = nll_loss(next_obs, means, logstds) if FLAGS.prob_model else mse_loss(pred, next_obs)
    if not FLAGS.no_learn_reward:
      assert reward_pred.ndim == reward.ndim  # no undesired broadcasting
      nll += ((reward_pred - reward) ** 2).mean()
    return nll

  def loss_vep(self, params_T, aux_params, batch, rng):
    assert not FLAGS.prob_model
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
    assert next_obs.ndim == pred.ndim  # no undesired broadcasting
    nll = nll_loss(next_obs, means, logstds) if FLAGS.prob_model else mse_loss(pred, next_obs)
    
    # note that VFs are random
    params_V = stop_gradient(aux_params)
    next_V = self.V.net.apply(params_V, next_obs)
    pred_V = self.V.net.apply(params_V, pred)
    l = 0
    for i in range(FLAGS.num_ensemble_vep):
      l += jnp.mean((next_V[i] - pred_V[i])**2)

    if not FLAGS.no_learn_reward:
      assert reward_pred.ndim == reward.ndim  # no undesired broadcasting
      l += ((reward_pred - reward) ** 2).mean()
    aux_out = AuxOut(None, None, nll)
    return l, aux_out

  def constraint_func(self, params_T, params_Q, replay, rng, target_params_Q):
    '''Parameterized by T function giving grad_Q (Bellman-error) = 0 constraint.
    '''
    replay_model, _ = self.batch_real_to_model(params_T, replay, rng)
    grads, aux_out = jax.grad(self.loss_Q, has_aux=True)(
      params_Q, target_params_Q, replay_model)
    return grads

  def fwd_solver(self, constraint_func, 
    params_Q, target_params_Q, opt_state_Q, params_T, replay, rng):
    """Get Q_* satisfying the constraint (approximately). 
    """
    replay_model, nll = self.batch_real_to_model(params_T, replay, rng)
    
    if FLAGS.no_warm:
      target_params_Q = params_Q
    if not FLAGS.warm_opt:
      opt_state_Q = self.Q.opt.init(params_Q)
    
    for i in range(FLAGS.num_Q_steps):
      updout = self.update_step(
        params_Q, target_params_Q, opt_state_Q, None, replay_model, 'sql')
      params_Q, opt_state_Q = updout.params_Q, updout.opt_state_Q
      target_params_Q = soft_update_params(FLAGS.tau, params_Q, target_params_Q)

    Sol = namedtuple('Sol', 
      'params_Q loss_Q vals_Q grad_norm_Q entropy_Q target_params_Q opt_state_Q next_obs_nll')
    return Sol(params_Q, updout.loss_Q, updout.vals_Q, updout.grad_norm_Q, 
      updout.entropy_Q, target_params_Q, opt_state_Q, nll)

  def loss_omd(self, params_T, aux_params, batch, replay):
    fwd_solver = lambda constraint_func, params_Q, params_T, replay, rng: self.fwd_solver(
      constraint_func, params_Q, aux_params.target_params_Q, aux_params.opt_state_Q, params_T, replay, rng)
    if FLAGS.no_warm:
      params_Q = self.Q.net.init(aux_params.rng, replay[0])
    else:
      params_Q = aux_params.params_Q

    # Update params_Q on a batch w.r.t. loss yielded by params_T
    sol = root_solve(self.constraint_func, params_Q, 
      params_T, replay, aux_params.rng, (fwd_solver,))
    
    return self.loss_Q(sol.params_Q, sol.target_params_Q, replay)[0], sol
  
  @partial(jax.jit, static_argnums=(0,6))
  @chex.assert_max_traces(n=1)
  def update_step(self, params, aux_params, opt_state, batch, replay, loss_type):
    if loss_type == 'sql':
      (value, aux_out), grads = value_and_grad(self.loss_Q, has_aux=True)(
        params, aux_params, replay)
      updates, opt_state = self.Q.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_Q params_Q opt_state_Q grads_Q grad_norm_Q vals_Q entropy_Q')
      return UpdOut(value, new_params, opt_state, grads, tree_norm(grads), 
        aux_out.vals_Q, aux_out.entropy_Q)

    elif loss_type == "mle":
      value, grads = value_and_grad(self.loss_mle)(params, replay, aux_params.rng)
      updates, opt_state = self.T.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_T params_T opt_state_T')
      return UpdOut(value, new_params, opt_state)

    elif loss_type == "omd":
      (value, aux_out), grads = value_and_grad(self.loss_omd, has_aux=True)(
        params, aux_params, batch, replay)
      updates, opt_state = self.T.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_T params_T opt_state_T loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')
      return UpdOut(value, new_params, opt_state, aux_out.loss_Q, 
        aux_out.vals_Q, aux_out.grad_norm_Q, aux_out.entropy_Q, aux_out.params_Q, 
        aux_out.target_params_Q, aux_out.opt_state_Q, aux_out.next_obs_nll)

    elif loss_type == "vep":
      (value, aux_out), grads = value_and_grad(self.loss_vep, has_aux=True)(
          params, aux_params.params_Q, replay, aux_params.rng)
      updates, opt_state = self.T.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_T params_T opt_state_T next_obs_nll')
      return UpdOut(value, new_params, opt_state, aux_out.next_obs_nll)

  def update(self, replay_buffer):
    replay = replay_buffer.sample(FLAGS.batch_size)
    
    if FLAGS.agent_type == 'omd':
      if FLAGS.no_warm:
        aux_params = AuxP(None, None, None, next(self.rngs))
      else:
        aux_params = AuxP(self.params_Q, self.target_params_Q, self.opt_state_Q, 
          next(self.rngs))
      updout = self.update_step(self.params_T, aux_params, self.opt_state_T, 
        None, replay, FLAGS.agent_type)
      self.params_Q, self.opt_state_Q = updout.params_Q, updout.opt_state_Q
      self.target_params_Q = updout.target_params_Q
      self.params_T, self.opt_state_T = updout.params_T, updout.opt_state_T
      return {'loss_T': updout.loss_T.item(), 
              'vals_Q': updout.vals_Q.item(), 
              'loss_Q': updout.loss_Q.item(), 
              'grad_norm_Q': updout.grad_norm_Q.item(), 
              'entropy_Q': updout.entropy_Q.item(),
              'next_obs_nll': updout.next_obs_nll.item()}

    elif FLAGS.agent_type == 'mle':
      for i in range(FLAGS.num_T_steps):
        aux_params = AuxP(None, None, None, next(self.rngs))
        replay = replay_buffer.sample(FLAGS.batch_size)
        updout_T = self.update_step(self.params_T, aux_params, self.opt_state_T, 
          None, replay, 'mle')
        self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

      for i in range(FLAGS.num_Q_steps):
        replay = replay_buffer.sample(FLAGS.batch_size)
        replay_model, nll = self.batch_real_to_model(self.params_T, replay, next(self.rngs))
        updout_Q = self.update_step(self.params_Q, self.target_params_Q, 
          self.opt_state_Q, None, replay_model, 'sql')    
        self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
        self.target_params_Q = soft_update_params(
          FLAGS.tau, self.params_Q, self.target_params_Q)

      return {'loss_T': updout_T.loss_T.item(), 
              'vals_Q': updout_Q.vals_Q.item(), 
              'loss_Q': updout_Q.loss_Q.item(), 
              'grad_norm_Q': updout_Q.grad_norm_Q.item(), 
              'entropy_Q': updout_Q.entropy_Q.item()}

    elif FLAGS.agent_type == 'vep':
      for i in range(FLAGS.num_T_steps):
        aux_params = AuxP(self.params_V, None, None, next(self.rngs))
        replay = replay_buffer.sample(FLAGS.batch_size)
        updout_T = self.update_step(self.params_T, aux_params, self.opt_state_T, 
          None, replay, 'vep')
        self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

      for i in range(FLAGS.num_Q_steps):
        replay = replay_buffer.sample(FLAGS.batch_size)
        replay_model, nll = self.batch_real_to_model(self.params_T, replay, next(self.rngs))
        updout_Q = self.update_step(self.params_Q, self.target_params_Q, 
          self.opt_state_Q, None, replay_model, 'sql')    
        self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
        self.target_params_Q = soft_update_params(
          FLAGS.tau, self.params_Q, self.target_params_Q)
      
      return {'loss_T': updout_T.loss_T.item(), 
              'vals_Q': updout_Q.vals_Q.item(), 
              'loss_Q': updout_Q.loss_Q.item(), 
              'grad_norm_Q': updout_Q.grad_norm_Q.item(), 
              'entropy_Q': updout_Q.entropy_Q.item(),
              'next_obs_nll': updout_T.next_obs_nll.item()}

  def save(self, agent_path):
    pickle.dump([self.params_Q, self.target_params_Q, self.params_T], 
                open(agent_path, 'wb'))
  
  def load(self, agent_path):
    self.params_Q, self.target_params_Q, self.params_T = pickle.load(
      open(agent_path, "rb"))
