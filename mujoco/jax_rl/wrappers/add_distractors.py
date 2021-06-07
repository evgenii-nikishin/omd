import gym
import numpy as np
from gym.spaces import Box


class AddDistractors(gym.ObservationWrapper):
    def __init__(self, env, dim_distract=0):
        super().__init__(env)

        assert isinstance(self.observation_space, Box)

        self.dim_distract = dim_distract

        obs_space = self.observation_space

        self.observation_space = Box(
            np.concatenate((obs_space.low, [-np.inf] * dim_distract)),
            np.concatenate((obs_space.high, [np.inf] * dim_distract)),
            (obs_space.shape[0] + dim_distract, ))

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (observation, np.random.randn(self.dim_distract)))
