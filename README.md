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
 
## Description   
What it does   

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
Next, navigate to any file and run it.   
```bash
# module folder
cd neural_clbf

# run module
python TODO.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from pytorch_lightning import Trainer

from neural_clbf.datasets.TODO import TODO
from neural_clbf.TODO import TODO

# model
model = TODO()

# data
train, val, test = TODO()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
