import unittest
import gymnasium as gym
import numpy as np
from noisyenv.wrappers import (
    RandomMixupObservation, RandomDropoutObservation, RandomNormalNoisyObservation, RandomUniformNoisyObservation,
    RandomUniformScaleObservation, RandomUniformScaleReward, RandomUniformNoisyReward, RandomNormalNoisyReward
)

ENV_ID = 'CartPole-v1'
SEED = 333


class BaseNoiseTest:
    NoiseClass = None

    def test_no_noise(self):
        env = gym.make(ENV_ID)
        wrapped_env = RandomDropoutObservation(gym.make(ENV_ID), noise_rate=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        for i in range(10):
            action = wrapped_env.action_space.sample()

            wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(
                action)
            obs, reward, terminated, truncated, info = env.step(action)

            np.testing.assert_almost_equal(obs, wrapped_obs)
            np.testing.assert_almost_equal(reward, wrapped_reward)
            np.testing.assert_almost_equal(terminated, wrapped_terminated)
            np.testing.assert_almost_equal(truncated, wrapped_truncated)


class TestRandomMixupObservation(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomMixupObservation

    def test_init(self):
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=0.1, factor=0.3)
        self.assertEqual(wrapped_env.noise_rate, 0.1)
        self.assertEqual(wrapped_env.factor, 0.3)
        self.assertIsNone(wrapped_env._last_observation)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, factor=0.5)
        observation1, *_ = env.reset(seed=SEED)
        wrapped_observation1, *_ = wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()

        observation2, *_ = env.step(action)
        wrapped_observation2, *_ = wrapped_env.step(action)

        expected_observation = 0.5 * observation2 + 0.5 * observation1
        np.testing.assert_array_equal(wrapped_observation2, expected_observation)
        np.testing.assert_array_equal(wrapped_observation1, observation1)


class TestRandomDropoutObservation(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomDropoutObservation

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, p=0.5)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.p, 0.5)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, p=1.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)
        for i in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            np.testing.assert_almost_equal(np.mean(obs), 0.0)


class TestRandomNormalNoisyObservation(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomNormalNoisyObservation

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, loc=1.0, scale=0.5)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.loc, 1.0)
        np.testing.assert_equal(wrapped_env.scale, 0.5)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, loc=1.0, scale=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()

        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(np.mean(wrapped_obs - obs), 1.0)
        np.testing.assert_almost_equal(reward, wrapped_reward)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)


class TestRandomUniformNoisyObservation(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomUniformNoisyObservation

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, low=-1.0, high=1.0)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.low, -1.0)
        np.testing.assert_equal(wrapped_env.high, 1.0)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=1.0, high=1.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()

        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(np.mean(wrapped_obs - obs), 1.0)
        np.testing.assert_almost_equal(reward, wrapped_reward)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)


class TestRandomUniformScaleObservation(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomUniformScaleObservation

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, low=-1.0, high=1.0)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.low, -1.0)
        np.testing.assert_equal(wrapped_env.high, 1.0)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=0.0, high=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(np.mean(wrapped_obs), 0.0)
        np.testing.assert_almost_equal(reward, wrapped_reward)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)

        # Test with 1.0
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=1.0, high=1.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(wrapped_obs, obs)


class TestRandomUniformScaleReward(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomUniformScaleReward

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, low=-1.0, high=1.0)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.low, -1.0)
        np.testing.assert_equal(wrapped_env.high, 1.0)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=0.0, high=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(obs, wrapped_obs)
        np.testing.assert_almost_equal(wrapped_reward, 0.0)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)

        # Test with 1.0
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=1.0, high=1.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(reward, wrapped_reward)


class TestRandomUniformNoisyReward(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomUniformNoisyReward

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, low=-1.0, high=1.0)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.low, -1.0)
        np.testing.assert_equal(wrapped_env.high, 1.0)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=0.0, high=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(obs, wrapped_obs)
        np.testing.assert_almost_equal(reward, wrapped_reward)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)

        # Test with 1.0
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, low=1.0, high=1.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(np.mean(wrapped_reward - reward), 1.0)


class TestRandomNormalNoisyReward(BaseNoiseTest, unittest.TestCase):
    NoiseClass = RandomNormalNoisyReward

    def test_init(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(env, noise_rate=0.1, loc=1.0, scale=0.5)
        np.testing.assert_equal(wrapped_env.noise_rate, 0.1)
        np.testing.assert_equal(wrapped_env.loc, 1.0)
        np.testing.assert_equal(wrapped_env.scale, 0.5)

    def test_observation(self):
        env = gym.make(ENV_ID)
        wrapped_env = self.NoiseClass(gym.make(ENV_ID), noise_rate=1.0, loc=1.0, scale=0.0)
        env.reset(seed=SEED)
        wrapped_env.reset(seed=SEED)

        action = wrapped_env.action_space.sample()

        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_almost_equal(np.mean(wrapped_reward - reward), 1.0)
        np.testing.assert_almost_equal(obs, wrapped_obs)
        np.testing.assert_almost_equal(terminated, wrapped_terminated)
        np.testing.assert_almost_equal(truncated, wrapped_truncated)


if __name__ == '__main__':
    unittest.main()
