# CPU/GPU performance

1. Copy the GPU executable to your home directory and run the test problem
```
$ cp src/problems/HydroBlast3D/test_hydro3d_blast ~/gpu-workshop-exercises/blast_gpu
$ cd ~/gpu-workshop-exercises
$ mpirun ./blast_gpu quokka/tests/blast_32.in
```
Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

2. CPU version

Now let's see how fast Quokka running on CPUs is. First, we have to re-compile Quokka for CPUs only:

```
$ cd /tmp/quokka
$ mkdir build_cpu
$ cd build_cpu
$ cmake .. -DAMReX_GPU_BACKEND=NONE -DAMReX_SPACEDIM=3
$ make -j16 test_hydro3d_blast
```

Then we will run copy the executable and run Quokka on 16 CPU cores:
```
$ cp src/problems/HydroBlast3D/test_hydro3d_blast ~/gpu-workshop-exercises/blast_cpu
$ cd ~/gpu-workshop-exercises
$ mpirun --oversubscribe -np 16 ./blast_cpu quokka/tests/blast_32.in
```

Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

How much faster is running on 1 GPU compared to running on 16 CPU cores?
Discuss in your group whether this is a fair way to compare CPU versus GPU performance. Consult the OzStar/NT documentation for reference.

## Collective discussion
