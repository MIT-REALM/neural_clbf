<div align="center">    
 
# Nonlinear Control Using Neural Lyapunov-Barrier Functions

[![Conference](http://img.shields.io/badge/TODO-1111-ff5e24.svg)](https://arxiv.org/)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/dawsonc/neural_clbf/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>

This repository includes code accompany several REALM papers (see below for references). 

## How to run

First, install Git LFS if you have not already
```bash
sudo apt-get install git-lfs
git lfs install
```

Then, clone the repository and install dependencies   
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
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
