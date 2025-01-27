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
$ mkdir build_gpu
$ cd build_gpu
$ cmake .. -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3
$ make -j16 test_hydro3d_blast
```
This should take about 5 minutes.

6. Copy the executable to your home directory and run the test problem
```
$ cp src/problems/HydroBlast3D/test_hydro3d_blast ~/gpu-workshop-exercises/blast_gpu
$ cd ~/gpu-workshop-exercises
$ mpirun ./blast_gpu quokka/tests/blast_32.in
```
Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

7. CPU version

Now let's see how fast Quokka running on CPUs is:

```
$ cd /tmp/quokka
$ mkdir build_cpu
$ cd build_cpu
$ cmake .. -DAMReX_GPU_BACKEND=NONE -DAMReX_SPACEDIM=3
$ make -j16 test_hydro3d_blast
```

Then we will run Quokka on 16 CPU cores:
```
$ cp src/problems/HydroBlast3D/test_hydro3d_blast ~/gpu-workshop-exercises/blast_cpu
$ cd ~/gpu-workshop-exercises
$ mpirun --oversubscribe -np 16 ./blast_cpu quokka/tests/blast_32.in
```

Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

How much faster is running on 1 GPU compared to running on 16 CPU cores?
Discuss in your group whether this is a fair way to compare CPU versus GPU performance. Consult the OzStar/NT documentation for reference.

## Collective discussion
