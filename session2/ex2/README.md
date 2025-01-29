# Comparing CPU/GPU performance, Part 2

## Exercise

1. Modify the parameters of the problem
Copy the file `quokka/tests/blast_32.in` to `quokka/tests/blast_128.in`. Edit the new file to have a grid size of 128 cells in each dimension. You can do this by changing the line that start with "amr.n_cell" to read:
```
amr.n_cell          = 128 128 128
```

2. Run the new test problem
```
$ mpirun ./blast_gpu quokka/tests/blast_128.in max_timesteps=500 plotfile_prefix=gpu128_plt
```
Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

3. CPU version

Then run Quokka on 16 CPU cores:
```
$ mpirun --oversubscribe -np 16 ./blast_cpu quokka/tests/blast_128.in max_timesteps=500 plotfile_prefix=cpu128_plt
```

Note the time it takes to run. This is printed to the terminal output (look for "elapsed time").

## Discussion
Why does this simulation take longer to run as compared to the previous simulation?

How many cells are used in this simulation? How many cells are updated per second for the GPU and CPU versions, respectively?

How much faster is running on 1 GPU compared to running on 16 CPU cores?

Discuss and form a hypothesis as to why this simulation shows a greater performance advantage on the GPU compared to the previous simulation.

## Collective discussion
