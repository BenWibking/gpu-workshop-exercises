#!/bin/sh

## environment setup for nt.swin.edu.au

## set up aliases
## system time limit is 1 hour
alias start_gpu_job='sinteractive --time=1:00:00 --mem=120g --cpus-per-task=16 --gres=gpu:1 --tmp=500G'
alias download_quokka='git clone --recursive https://github.com/quokka-astro/quokka.git'

## set up environment modules
module load gcc/12.3.0
module load cmake/3.26.3
module load cuda/12.6.0
module load openmpi/4.1.5
module load hdf5/1.14.0
module load python/3.11.3

python3 -m venv ~/gpu-workshop-exercises/pyenv_gpu
source ~/gpu-workshop-exercises/pyenv_gpu/bin/activate
pip install --upgrade pip
pip install yt ipython jupyter
