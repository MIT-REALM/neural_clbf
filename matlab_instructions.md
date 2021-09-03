# How to install this project for use with MATLAB

NOTE: you need to be running MATLAB 2020b or later. I'm testing on 2021a. Also, make sure Gurobi is installed and activated.

```
# clone project   
git clone https://github.com/dawsonc/neural_clbf

# install project   
cd neural_clbf
conda create --name neural_clbf_matlab python=3.8
conda activate neural_clbf_matlab
pip install -e .   
pip install -r requirements.txt
```

Verify that the instalation worked by running `pytest .`

Get the path where python is installed by running `which python`

Launch MATLAB, then run the following in the MATLAB prompt
```
pe = pyenv("Version","<path to python>", "ExecutionMode", "OutOfProcess")
```

Verify that MATLAB has found Python by running `py.list`. You should see an output of `Python list with no properties. []`. Next try running 

```
torch = py.importlib.import_module('torch');
```

You should only need to do this once.

# Using the Neural CBF QP from MATLAB

First, import the `neural_clbf` module into matlab.
```
neural_clbf = py.importlib.import_module('neural_clbf');
```

