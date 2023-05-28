.. noisyenv Documentation documentation master file, created by
   sphinx-quickstart on Sun May 28 21:54:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to noisyenv's documentation!
==================================================

This package contains a set of generic wrappers designed to augment RL environments with noise and encourage agent exploration and improve training data diversity which are applicable to a broad spectrum of RL algorithms and environments. For more details, please refer to our paper: https://arxiv.org/abs/2305.02882.

Note that this package has been developed for the new step and reset API introduced in OpenAI Gym v26 and Gymnasium v26. Use the `gymnasium.wrappers.EnvCompatibility` wrapper to update old environments for compatibility.

.. code-block:: bash

   $ pip install noisyenv


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

