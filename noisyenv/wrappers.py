import gymnasium as gym
import numpy as np


class RandomMixupObservation(gym.ObservationWrapper):
    """Adds random mixup noise to the observations of the environment.

    The mixup noise is a convex combination of the current observation and the previous observation.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomMixupObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomMixupObservation(env, noise_rate=0.1, factor=0.3)
    """

    def __init__(self, env, noise_rate=0.01, factor=0.5):
        """Initializes the :class:`RandomMixupObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of applying mixup to the observation each step.
                Defaults to 0.01.
            factor (float, optional): The mixup factor (factor * observation + (1 - factor) * last_observation).
                Defaults to 0.5.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.factor = factor
        self._last_observation = None

    def reset(self, **kwargs):
        """Resets the environment, returning a potentially modified observation using :meth:`self.observation`."""
        obs, info = super().reset(**kwargs)
        self._last_observation = obs
        return obs, info

    def observation(self, observation):
        """Returns the potentially modified observation."""
        if (np.random.rand() <= self.noise_rate) and (self._last_observation is not None):
            observation = self.factor * observation + (1 - self.factor) * self._last_observation

        self._last_observation = observation

        return observation


class RandomDropoutObservation(gym.ObservationWrapper):
    """Applies dropout to the observations of the environment.

    Dropout randomly replaces elements of the observation with 0 with probability p.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomDropoutObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomDropoutObservation(env, noise_rate=0.1, p=0.5)
    """

    def __init__(self, env, noise_rate=0.01, p=0.1):
        """Initializes the :class:`RandomDropoutObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of applying dropout to the observation each step.
                Defaults to 0.01.
            p (float, optional): The probability of replacing an element of the observation with a 0. Defaults to 0.1.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.p = p

    def observation(self, observation):
        """Returns the potentially modified observation."""
        if (np.random.rand() <= self.noise_rate):
            observation *= np.random.binomial(np.ones(observation.shape).astype(int), p=(1 - self.p))
        return observation


class RandomNormalNoisyObservation(gym.ObservationWrapper):
    """Adds random Normal noise to the observations of the environment.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomNormalNoisyObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomNormalNoisyObservation(env, noise_rate=0.1, loc=0.0, scale=0.1)
    """

    def __init__(self, env, noise_rate=0.01, loc=0.0, scale=0.01):
        """Initializes the :class:`RandomNormalNoisyObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the observation each step.
                Defaults to 0.01.
            loc (float, optional): Mean ("centre") of the noise distribution.
                Defaults to 0.0.
            scale (float, optional): Standard deviation (spread or "width") of the noise distribution.
                Must be non-negative. Defaults to 0.01.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.loc = loc
        self.scale = scale

    def observation(self, observation):
        """Returns the potentially modified observation."""
        if np.random.rand() <= self.noise_rate:
            observation += np.random.normal(loc=self.loc, scale=self.scale, size=observation.shape)
        return observation


class RandomUniformNoisyObservation(gym.ObservationWrapper):
    """Adds random Uniform noise to the observations of the environment.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomUniformNoisyObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomUniformNoisyObservation(env, noise_rate=0.1, low=-0.1, high=0.1)
    """

    def __init__(self, env, noise_rate=0.01, low=-0.1, high=0.1):
        """Initializes the :class:`RandomUniformNoisyObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the observation each step.
                Defaults to 0.01.
            low (float, optional): Lower boundary of the noise distribution.
                Defaults to -0.1.
            high (float, optional): Upper boundary of the noise distribution.
                Defaults to 0.1.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.low = low
        self.high = high

    def observation(self, observation):
        """Returns the potentially modified observation."""
        if np.random.rand() <= self.noise_rate:
            observation += np.random.uniform(low=self.low, high=self.high, size=observation.shape)
        return observation


class RandomUniformScaleObservation(gym.ObservationWrapper):
    """Scales the observations by random Uniform noise.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomUniformScaleObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomUniformScaleObservation(env, noise_rate=0.1, low=0.9, high=1.1)
    """

    def __init__(self, env, noise_rate=0.01, low=0.9, high=1.1, size=1):
        """Initializes the :class:`RandomUniformScaleObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of applying noise to the observation each step.
                Defaults to 0.01.
            low (float, optional): Lower boundary of the noise distribution.
                Defaults to 0.9.
            high (float, optional): Upper boundary of the noise distribution.
                Defaults to 1.1.
            size (int, optional): The number of scaling factors to sample.
                If None then size is equal to observation.shape. Defaults to 1.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.low = low
        self.high = high
        self.size = size

    def observation(self, observation):
        """Returns the potentially modified observation."""
        if np.random.rand() <= self.noise_rate:
            if self.size is None:
                size = observation.shape
            else:
                size = self.size
            observation *= np.random.uniform(low=self.low, high=self.high, size=size)
        return observation


class RandomUniformScaleReward(gym.RewardWrapper):
    """Scales the rewards by random Uniform noise.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomUniformScaleReward
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomUniformScaleReward(env, noise_rate=0.1, low=0.9, high=1.1)
    """

    def __init__(self, env, noise_rate=0.01, low=0.9, high=1.1):
        """Initializes the :class:`RandomUniformScaleReward` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of applying noise to the reward each step.
                Defaults to 0.01.
            low (float, optional): Lower boundary of the noise distribution.
                Defaults to 0.9.
            high (float, optional): Upper boundary of the noise distribution.
                Defaults to 1.1.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.low = low
        self.high = high

    def reward(self, reward):
        """Returns the potentially modified reward."""
        if np.random.rand() <= self.noise_rate:
            reward *= np.random.uniform(self.low, self.high)
        return reward


class RandomUniformNoisyReward(gym.RewardWrapper):
    """Adds random Uniform noise to the rewards.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomUniformNoisyReward
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomUniformNoisyReward(env, noise_rate=0.1, low=-0.1, high=0.1)
    """

    def __init__(self, env, noise_rate=0.01, low=-0.01, high=0.01):
        """Initializes the :class:`RandomUniformNoisyReward` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the reward each step.
                Defaults to 0.01.
            low (float, optional): Lower boundary of the noise distribution.
                Defaults to -0.1.
            high (float, optional): Upper boundary of the noise distribution.
                Defaults to 0.1.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.low = low
        self.high = high

    def reward(self, reward):
        """Returns the potentially modified reward."""
        if np.random.rand() <= self.noise_rate:
            reward += np.random.uniform(self.low, self.high)
        return reward


class RandomNormalNoisyReward(gym.RewardWrapper):
    """Adds random Normal noise to the rewards.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomNormalNoisyReward
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomNormalNoisyReward(env, noise_rate=0.1, scale=0.1)
    """

    def __init__(self, env, noise_rate=0.01, loc=0.0, scale=0.01):
        """Initializes the :class:`RandomNormalNoisyReward` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the reward each step.
                Defaults to 0.01.
            loc (float, optional): Mean ("centre") of the noise distribution.
                Defaults to 0.0.
            scale (float, optional): Standard deviation (spread or "width") of the noise distribution.
                Must be non-negative. Defaults to 0.01.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.loc = loc
        self.scale = scale

    def reward(self, reward):
        """Returns the potentially modified reward."""
        if np.random.rand() <= self.noise_rate:
            reward += np.random.normal(self.loc, self.scale)
        return reward
