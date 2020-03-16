#!/bin/bash

# Setup env
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cs236781-project
echo "hello from $(python --version) in $(which python)"

python train.py
