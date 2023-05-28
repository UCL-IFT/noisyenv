noisyenv
========

This package contains a set of generic wrappers designed to augment RL environments with noise and encourage agent exploration and improve training data diversity which are applicable to a broad spectrum of RL algorithms and environments. For more details, please refer to our paper: https://arxiv.org/abs/2305.02882.

Note that this package has been developed for the new step and reset API introduced in OpenAI Gym v26 and Gymnasium v26. Use the `gymnasium.wrappers.EnvCompatibility` wrapper to update old environments for compatibility.

.. toctree::
   :maxdepth: 4

   noisyenv