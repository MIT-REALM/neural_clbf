#!/bin/bash

for i in {1..20}
do
    python neural_clbf/training/train_cartpole.py --n-epochs 50 --track --wandb-group cartpole
done
