# Optimal Model Design for Reinforcement Learning

This repository contains JAX code for the paper 

[**Control-Oriented Model-Based Reinforcement Learning with Implicit Differentiation**](https://arxiv.org/abs/2106.03273)

by [Evgenii Nikishin](http://evgenii-nikishin.github.io/), [Romina Abachi](https://ca.linkedin.com/in/romina-abachi-44915bbb), [Rishabh Agarwal](https://agarwl.github.io/), and [Pierre-Luc Bacon](http://pierrelucbacon.com/).


# Summary

Model based reinforcement learning typically trains the dynamics and reward functions by minimizing the error of predictions.
The error is only a proxy to maximizing the sum of rewards, the ultimate goal of the agent, leading to [the objective mismatch](https://arxiv.org/abs/2002.04523).
We propose an end-to-end algorithm called *Optimal Model Design* (OMD) that optimizes the returns directly for model learning.
OMD leverages the implicit function theorem to optimize the model parameters and forms the following computational graph:

<p align="center">
  <img src="https://user-images.githubusercontent.com/14283069/120944292-bde92500-c701-11eb-9695-17378d26440f.png" width=500>
</p>

Please cite our work if you find it useful in your research:
```latex
@article{nikishin2021control,
  title={Control-Oriented Model-Based Reinforcement Learning with Implicit Differentiation},
  author={Nikishin, Evgenii and Abachi, Romina and Agarwal, Rishabh and Bacon, Pierre-Luc},
  journal={arXiv preprint arXiv:2106.03273},
  year={2021}
}
```

# Installation

We assume that you use Python 3. To install the necessary dependencies, run the following commands: 

```bash
1. virtualenv ~/env_omd
2. source ~/env_omd/bin/activate
3. pip install -r requirements.txt
```

To use JAX with GPU, follow [the official instructions](https://github.com/google/jax#installation).
To install MuJoCo, check [the instructions](https://github.com/openai/mujoco-py/#install-and-use-mujoco-py).


# Run

For historical reasons, the code is divided into 3 parts.

## Tabular

All results for the tabular experiments could be reproduced by running the `tabular.ipynb` notebook.

To open the notebook in Google Colab, use [this link](https://colab.research.google.com/github/evgenii-nikishin/omd/blob/main/tabular.ipynb).

## CartPole

To train the OMD agent on CartPole, use the following commands:

```bash
cd cartpole
python train.py --agent_type omd
```

We also provide the implementation of the corresponding MLE and VEP baselines. To train the agents, change the `--agent_type` flag to `mle` or `vep`. 

## MuJoCo

To train the OMD agent on MuJoCo HalfCheetah-v2, use the following commands:

```bash
cd mujoco
python train.py --config.algo=omd
```

To train the MLE baseline, change the `--config.algo` flag to `mle`. 


# Acknowledgements

* Tabular experiments are based on the code from [the library for fixed points in JAX](https://github.com/gehring/fax)
* Code for MuJoCo is based on [the implementation of SAC in JAX](https://github.com/ikostrikov/jax-rl/)
* Code for CartPole reuses parts of [the SAC implementation in PyTorch](https://github.com/denisyarats/pytorch_sac)
* For experimentation, we used a moditication of [the slurm runner](https://github.com/willwhitney/exploration-reimplementation/blob/master/runner.py)
