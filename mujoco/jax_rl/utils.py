from typing import Optional

import gym
from gym.wrappers import RescaleAction

from jax_rl import wrappers


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             dim_distract: int = 0) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    assert env_name in env_ids

    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = wrappers.VideoRecorder(env, save_folder=save_folder)

    if dim_distract > 0:
        env = wrappers.AddDistractors(env, dim_distract=dim_distract)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
