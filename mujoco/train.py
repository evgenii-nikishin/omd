import os
import random

import jax
import numpy as np
import tqdm
from absl import app, flags
from tensorboardX import SummaryWriter

from jax_rl.agents import OMDLearner
from jax_rl.datasets import ReplayBuffer
from jax_rl.evaluation import evaluate
from jax_rl.utils import make_env
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Custom description string added to save_dir.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('dim_distract', 0, 'Number of distracting states.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('no_jit', False, 'Disable JIT (will make code slower).')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/omd_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    jax.config.update('jax_disable_jit', FLAGS.no_jit)

    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name,
                   FLAGS.seed,
                   video_train_folder,
                   dim_distract=FLAGS.dim_distract)
    eval_env = make_env(FLAGS.env_name,
                        FLAGS.seed + 42,
                        video_eval_folder,
                        dim_distract=FLAGS.dim_distract)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    outer_steps = kwargs.pop('outer_steps')

    if algo == 'omd' or algo == 'mle':
        # MLE is implemented within
        agent = OMDLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis],
                           **kwargs,
                           algo=algo)
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            for _ in range(outer_steps):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
