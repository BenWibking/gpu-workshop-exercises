# Getting started with QUOKKA

## Goals for this session

* Understand how to compare performance GPU/CPU
* Understand how problem size (cells/GPU) affects performance
* Understand how grid size affects performance
* Understand how to visualize output in a Jupyter Notebook with yt

## Preliminary setup

1. Load the Bash profile:
```
$ . nt.profile
```

2. Download QUOKKA:
```
$ download_quokka
```

3. Start an interative job on a GPU node:
```
$ start_gpu_job
```

4. Copy the `quokka` directory to `/tmp`:
```
$ cp -r quokka /tmp
```

5. Compile QUOKKA:
```
$ cd /tmp/quokka
$ mkdir build
$ cd build
$ cmake .. -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3
$ make -j32 test_hydro3d_blast
```
This should take about 5 minutes.

6. Copy the executable to your home directory and run the test problem
```
$ cp src/problems/HydroBlast3D/test_hydro3d_blast ~/gpu-workshop-exercises/
$ cd ~/gpu-workshop-exercises
$ mpirun ./test_hydro3d_blast quokka/tests/blast_32.in
```
This should take less than 1 minute to run.
