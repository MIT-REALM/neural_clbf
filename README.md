<div align="center">

# Nonlinear Control Using Neural Lyapunov-Barrier Functions

[![Conference](https://img.shields.io/badge/CoRL%20'21-Accepted-success)](https://openreview.net/forum?id=8K5kisAnb_p)
[![Paper](https://img.shields.io/badge/RAL%20'21-Accepted-success)](https://ieeexplore.ieee.org/abstract/document/9676477)
</div>

This repository includes code accompany several REALM papers (see below for references). Specifically, it hosts our framework for using neural networks to learn certificates (usually Lyapunov, Barrier, or Lyapunov-Barrier functions) to robustly control nonlinear dynamical systems. Papers that reference this code include:

- ["Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions" in CoRL 2021](https://openreview.net/forum?id=8K5kisAnb_p)
- ["Learning Safe, Generalizable Perception-Based Hybrid Control With Certificates" in RA-L & ICRA 2022](https://ieeexplore.ieee.org/abstract/document/9676477)
- 

Disclaimer: this code is research-grade, and should not be used in any production setting. It may contain outdated dependencies or security bugs, and of course we cannot guarantee the safety of our controllers on your specific autonomous system. If you have a particular application in mind, please reach out and we'll be happy to discuss with you.

## How to run

Clone the repository and install dependencies
```bash
# clone project
git clone https://github.com/dawsonc/neural_clbf

# install project
cd neural_clbf
conda create --name neural_clbf python=3.9
conda activate neural_clbf
pip install -e .
pip install -r requirements.txt
```

Next, make sure everything is installed correctly
```bash
pytest .
```

Training:
To setup port forwarding for TensorBoard:
`ssh -L 16006:127.0.0.1:6006 cbd@realm-01.mit.edu`

### A note on QP solvers

The default behavior is to train and evaluate using CVXPy to solve the relevant quadratic programs (QPs). CVXPy is free and enabled backprop through QP solutions via CVXPyLayers (which is why we use it for training), but you can get faster evaluations by using Gurobi instead (requires a license, but free for academics). Setting the `disable_gurobi` flag to `False` will enable Gurobi (this is required if you want to exactly reproduce the behavior we report in our papers, since we used Gurobi for those results, but it's optional if you just want to experiment with our code).

### Defining new examples

If you want to apply the Neural CLBF control method to your own problems, you need to start by defining a subclass of `neural_clbf.systems.ControlAffineSystem` for your control problem. For an example, see `neural_clbf.systems.InvertedPendulum`. This subclass should specify:

- The dynamics in control-affine form. Recall that control-affine means $dx/dt = f(x) + g(x)u$, so if $x$ is `(batch_size, n_dims)`, then $f$ is `(batch_size, n_dims)` and $g$ is `(batch_size, n_dims, n_controls)`. $f$ and $g$ should be implemented in the `_f` and `_g` functions, respectively.
- The parameters of the dynamics (in `validate_params`). These can include any parameters that have uncertainty you want to be robust to; for example, mass.
- The dimension of the system's state ('n_dims'), control input ('n_controls'), and the indices of any state dimensions that are angles ('angle_dims', so that we can normalize these angles appropriately).
- Limits on states (to define the training region, in `state_limits`) and controls (these are enforced in the controller, set in `control_limits`).
- Definitions of the safe, unsafe, and goal regions ('safe_mask', 'unsafe_mask', and 'goal_mask'). Each of these functions should behave as described in the docstring; the high-level is that these functions return a tensor of booleans, and each row is True if the corresponding row in the input is in the region.
- The goal point in `goal_point` and the control input needed to make that a fixed point `u_eq`.
- All `ControlAffineSystem` subclasses by default have a nominal controller (the controller that feeds into the CLBF filter) using LQR feedback gains obtained automatically by linearizing `_f` and `_g`. If you want a different nominal controller, you can define it by overriding `u_nominal`.

There are a few more advanced/experimental features you can explore if needed, but these are the basics that any new application should need.

Once you've defined your new `ControlAffineSystem` subclass, you can write a training script (start by copying the example in `neural_clbf.training.train_inverted_pendulum`). This script defines the scenarios for robust control, any experiments you want to run during training, the range of training data, and some hyperparameters for training. Run it and observe the results using TensorBoard :D!

### External dependencies

#### F16 Model
To install the F16 simulator (which is a GPL-licensed component and thus not distributed along with this code), you should also run
```
cd ..  # or wherever you want to put it
git clone git@github.com:dawsonc/AeroBenchVVPython.git
cd AeroBenchVVPython
pwd
```
Then go to `neural_clbf/setup/aerobench.py` and modify it to point to the path to the aerobench package.

### Citation

If you find this code useful in your own research, please cite our relevant papers:

A survey of neural certificate (CLF, CBF, ...) methods:
```
@ARTICLE{dawson2022survey,
  author={Dawson, Charles and Gao, Sicun and Fan, Chuchu},
  journal={IEEE Transactions on Robotics}, 
  title={Safe Control With Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods for Robotics and Control}, 
  year={2023},
  volume={},
  number={},
  pages={1-19},
  doi={10.1109/TRO.2022.3232542}}
```

Learning robust control Lyapunov-Barrier functions (rCLBFs) using neural networks:
```
@inproceedings{
  dawson2021safe,
  title={Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions},
  author={Charles Dawson and Zengyi Qin and Sicun Gao and Chuchu Fan},
  booktitle={5th Annual Conference on Robot Learning },
  year={2021},
  url={https://openreview.net/forum?id=8K5kisAnb_p}
}
```

Perception-based barrier functions for safe control from observations:
```
@ARTICLE{dawson2022perception,
  author={Dawson, Charles and Lowenkamp, Bethany and Goff, Dylan and Fan, Chuchu},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Safe,
  Generalizable Perception-Based Hybrid Control With Certificates},
  year={2022},
  volume={7},
  number={2},
  pages={1904-1911},
  doi={10.1109/LRA.2022.3141657}}
```
