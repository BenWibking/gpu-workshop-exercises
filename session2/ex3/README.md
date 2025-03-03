# Comparing CPU/GPU performance, Part 3

## Exercise

1. Modify the parameters of the problem
Copy the file `quokka/tests/blast_128.in` to `quokka/tests/blast_128_maxgrid128.in`.

Then change the parameters of the new file `quokka/tests/blast_128_maxgrid128.in` so that the simulation data is stored in a single grid of 128 cells in each dimension.

You can do this by changing the lines that start with `amr.max_grid_size` and `amr.blocking_factor` to read:
```
amr.max_grid_size   = 128   # at least 128 for GPUs
amr.blocking_factor = 128   # grid size must be divisible by this
```

2. Run the new test problem
```
$ mpirun ./blast_gpu quokka/tests/blast_128_maxgrid128.in max_timesteps=500 plotfile_prefix=gpu128_maxgrid128_plt
```
Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

(These grid parameters are not optimal on the CPU, so we will not run the CPU version.)

## Discussion
How much faster is this version of the simulation compared to the previous version?

How many cells are used in this simulation? How many cells are updated per second?

Discuss and form a hypothesis as to why this simulation is faster than before.

## Collective discussion
