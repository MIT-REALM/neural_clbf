<div align="center">    
 
# Nonlinear Control Using Neural Lyapunov-Barrier Functions

[![Conference](https://img.shields.io/badge/CoRL-Accepted-success)](https://openreview.net/forum?id=8K5kisAnb_p)
   
[![Arxiv](http://img.shields.io/badge/arxiv-eess.sy:2109.06697-B31B1B.svg)](https://arxiv.org/abs/2109.06697)

This repository contains the most up-to-date code for our CoRL paper, "Safe Nonlinear Control Using Robust Lyapunov-Barrier Functions." This repository contains many of the examples used in our paper, but some (such as the neural lander and 2D quadrotor with obstacles) can be found in [this repository](https://github.com/dawsonc/neural_clbf_experiments), which includes code that is less ready for prime time but may serve as a useful reference.

<!--  
Conference   
-->   
</div>
 
## Description
This repository contains our code for learning robust Lyapunov-style control certificates for safety and stability for nonlinear dynamical systems.

## How to run
First, install dependencies   
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

Once installed, training examples can be run using e.g. `python neural_clbf/training/train_single_track_car.py`, and pre-trained models can be evaluated using the scripts in the `neural_clbf/evaluation` directory. To run training on a remote server with port forwarding for TensorBoard, connect using `ssh -L 16006:127.0.0.1:6006 cbd@realm-01.mit.edu`. To view the training progress (including plots of simulated controller performance), run `tensorboard --logdir=logs` from a new terminal window.

If you're new to the codebase, try starting with these training examples, which should work with no tuning.
- `python neural_clbf/training/train_inverted_pendulum.py`: everyone's favorite toy problem! Can be completely visualized in 2D, and provides a good introduction to the `NeuralCLBFController` class and the `ControlAffineSystem` abstract base class, as implemented in `InvertedPendulum`.
- `python neural_clbf/training/train_kinematic_car.py` and `python neural_clbf/training/train_single_track_car.py`: same control architecture as the inverted pendulum, but with more complex dynamics. Both of these use robust controllers to track a-priori unknown paths using different car models.
- `python neural_clbf/training/train_linear_satellite.py`: trains a neural control barrier function (CBF), with no Lyapunov component, for satellite collision avoidance.

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

#### MATLAB Bridge for Robust MPC

```
cd "matlabroot/extern/engines/python"
python setup.py install
```


### Citation
If you find this code useful in your own research, please cite our corresponding paper.

```
@article{dawson_neural_clbf_2021,
  title={Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions},
  author={Charles Dawson, Zengyi Qin, Sicun Gao, Chuchu Fan},
  journal={5th Annual Conference on Robot Learning},
  year={2021}
}
```   
