import os
import gym
import numpy as np
import time
import pickle
from absl import app
from absl import flags

import jax
from jax.config import config
config.update("jax_enable_x64", True)

from logger import Logger
from omd import Agent
from replay_buffer import ReplayBuffer
from utils import *


FLAGS = flags.FLAGS
flags.DEFINE_string('exp', '', 'Custom description string added to out_dir')
flags.DEFINE_string('out_dir', 'out', 'Directory for output files')
flags.DEFINE_string('data_path', 'data/buf.pkl', 'Path to save buffer')
flags.DEFINE_string('agent_path', 'data/agent.pkl', 'Path to save agent')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Gym environment id')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('num_train_steps', 200000, 'Env steps num', lower_bound=0)
flags.DEFINE_integer('hidden_dim', 32, 'Size of hidden layers', lower_bound=1)
flags.DEFINE_integer('model_hidden_dim', 32, 'Model-specific', lower_bound=1)
flags.DEFINE_integer('batch_size', 256, 'Mini-batch samples', lower_bound=1)
flags.DEFINE_integer('init_steps', 1000, 'Steps before training', lower_bound=0)
flags.DEFINE_integer('eval_frequency', 1000, 'Agent evaluation', lower_bound=1)
flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency', lower_bound=1)
flags.DEFINE_integer('num_Q_steps', 1, 'Inner loop steps num', lower_bound=1)
flags.DEFINE_integer('num_T_steps', 1, 'Model steps per update', lower_bound=1)
flags.DEFINE_integer('num_ensemble_vep', 5, 'Value functions #', lower_bound=1)
flags.DEFINE_float('eps', 0.1, 'Random action probability')
flags.DEFINE_float('discount', 0.99, 'Sum of rewards discount factor')
flags.DEFINE_float('lr', 1e-3, '(Outer loop) learning rate')
flags.DEFINE_float('inner_lr', 3e-4, 'Inner loop learning rate')
flags.DEFINE_float('alpha', 0.01, 'Temperature')
flags.DEFINE_float('tau', 0.01, 'Target network update coefficient')
flags.DEFINE_boolean('save_buf', False, 'Save collected buffer with data')
flags.DEFINE_boolean('save_agent', False, 'Save the agent after training')
flags.DEFINE_boolean('hard', False, 'max vs logsumexp for Q learning')
flags.DEFINE_boolean('no_learn_reward', False, 'Use trainable or true rewards')
flags.DEFINE_boolean('prob_model', False, 'Gaussian vs deterministic next obs')
flags.DEFINE_boolean('no_warm', False, 'Not use previous Q* in the inner loop')
flags.DEFINE_boolean('warm_opt', False, 'Reuse inner loop optimizer statistics')
flags.DEFINE_boolean('no_double', False, 'Not use Double Q Learning')
flags.DEFINE_boolean('with_inv_jac', False, 
  'Replaces inverse Jacobian in implicit gradient with identity matrix')
flags.DEFINE_enum('agent_type', 'omd', ['omd', 'mle', 'vep'], 'Agent type')


def main(_):
  env = gym.make(FLAGS.env_name)
  eval_env = gym.make(FLAGS.env_name)

  for e in [eval_env, env]:
    e.seed(FLAGS.seed)
    e.action_space.seed(FLAGS.seed)
    e.observation_space.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  rngs = hk.PRNGSequence(FLAGS.seed)
  
  agent = Agent(env.observation_space, env.action_space)
  replay_buffer = ReplayBuffer(env.observation_space.shape, FLAGS.num_train_steps)

  FLAGS.out_dir = os.path.join(FLAGS.out_dir, str(FLAGS.seed))
  os.makedirs(FLAGS.out_dir, exist_ok=True)
  logger = Logger(FLAGS.out_dir, save_tb=False, agent=FLAGS.agent_type)
  
  step, episode = 0, 0
  episode_return, episode_step = 0, 0
  done = False
  obs = env.reset()
  action = agent.act(agent.params_Q, obs, next(rngs))
  
  start_time = time.time()
  print("Beginning of training")
  while step < FLAGS.num_train_steps:
    # evaluate agent periodically
    if step % FLAGS.eval_frequency == 0:
      eval_return = evaluate(agent, eval_env, next(rngs))
      logger.log('eval/episode', episode, step)
      logger.log('eval/episode_return', eval_return, step)
      logger.dump(step, ty='eval')

    # with epsilon exploration
    action = env.action_space.sample() if (np.random.rand() < FLAGS.eps or step < FLAGS.init_steps) else action.item()
    next_obs, reward, done, _ = env.step(action)

    done = float(done)
    # allow infinite bootstrap
    done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
    episode_return += reward

    replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
    
    obs = next_obs
    episode_step += 1
    step += 1

    if done:
      logger.log('train/episode', episode, step)
      logger.log('train/episode_return', episode_return, step)
      logger.log('train/duration', time.time() - start_time, step)

      obs = env.reset()
      done = False
      episode_return = 0
      episode_step = 0
      episode += 1

    action = agent.act(agent.params_Q, obs, next(rngs))
    
    if step >= FLAGS.init_steps:
      losses_dict = agent.update(replay_buffer)
      for k, v in losses_dict.items():
        logger.log('train/{}'.format(k), v, step)

      if (step % FLAGS.log_frequency) == 0:
        logger.dump(step, ty='train')
        if FLAGS.save_buf:
          replay_buffer.save(FLAGS.data_path)
        if FLAGS.save_agent:
          agent.save(FLAGS.agent_path)

  # final eval after training is done
  eval_return = evaluate(agent, eval_env, next(rngs))
  logger.log('eval/episode', episode, step)
  logger.log('eval/episode_return', eval_return, step)
  logger.dump(step, ty='eval')

  print("Done in {:.1f} minutes".format((time.time() - start_time)/60))

if __name__ == '__main__':
  app.run(main)
