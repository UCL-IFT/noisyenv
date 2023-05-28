# `noisyenv`: Simple Noisy Environment Augmentation for Reinforcement Learning

This package contains a set of generic wrappers designed to augment RL environments with noise and encourage agent exploration and improve training data diversity which are applicable to a broad spectrum of RL algorithms and environments. For more details, please refer to our paper: https://arxiv.org/abs/2305.02882.

Note that this package has been developed for the new step and reset API introduced in [OpenAI Gym v26](https://github.com/openai/gym/releases/tag/0.26.2) and [Gymnasium v26](https://gymnasium.farama.org/content/migration-guide/). Use the `gymnasium.wrappers.EnvCompatibility` wrapper to update old environments for compatibility. 

## Installation

```shell
pip install noisyenv
```

## Usage

```python
import gymnasium as gym
from noisyenv.wrappers import RandomUniformScaleReward

base_env = gym.make("HalfCheetah-v2")
env = RandomUniformScaleReward(env=base_env, noise_rate=0.01, low=0.9, high=1.1)

# And just use as you would normally
observation, info = env.reset(seed=333)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```


## Citing noisyenv
If you use `noisyenv` in your work, please cite our paper:

```bibtex
@misc{khraishi2023simple,
      title={Simple Noisy Environment Augmentation for Reinforcement Learning}, 
      author={Raad Khraishi and Ramin Okhrati},
      year={2023},
      eprint={2305.02882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```