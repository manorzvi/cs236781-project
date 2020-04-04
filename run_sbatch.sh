#!/bin/bash

# Setup env
source $HOME/anaconda3/etc/profile.d/conda.sh
#conda activate cs236781-project
conda activate cs236781-hw
echo "hello from $(python --version) in $(which python)"

python -c 'import torch; print(f"i can haz gpu? {torch.cuda.is_available()}")'
python train_gridsearch.py
