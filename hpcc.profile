#!/bin/sh

## environment setup for hpcc.msu.edu

## set up aliases
alias download_quokka='git clone --recursive https://github.com/quokka-astro/quokka.git'

## set up environment modules
module load CUDA/12.6.0

python3 -m venv ~/gpu-workshop-exercises/pyenv_gpu
source ~/gpu-workshop-exercises/pyenv_gpu/bin/activate
pip install --upgrade pip
pip install yt ipython jupyter
