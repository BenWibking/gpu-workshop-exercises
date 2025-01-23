#!/bin/sh

## environment setup for nt.swin.edu.au

## set up aliases
alias getgpu='sinteractive --time=1:00:00 --mem=128g --cpus-per-task=16 --gres=gpu:1'
alias getquokka='git clone --recursive https://github.com/quokka-astro/quokka.git'

## set up environment modules
module load gcc/12.3.0
module load cmake/3.26.3
module load cuda/12.6.0
module load openmpi/4.1.5
module load hdf5/1.14.0
