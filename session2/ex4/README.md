# Bonus: visualizing the output

## Running a full simulation

Let's re-run the last simulation, but let it run up to 10,000 timesteps:
```
$ mpirun ./blast_gpu quokka/tests/blast_128_maxgrid128.in plotfile_prefix=gpu128_maxgrid128_plt
```

## Visualization

Let's look at the simulations we ran.

Open the included Jupyter Notebook `Visualization.ipynb` by running

```
$ cd session2/ex4
$ jupyter notebook --no-browser
```

After a few seconds, you should see output similar to this:
```
    To access the server, open this file in a browser:
        file:///home/bwibking/.local/share/jupyter/runtime/jpserver-1977738-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=399225e03a3457569e526ae00a1f5c3442f768d6108fe680
        http://127.0.0.1:8888/tree?token=399225e03a3457569e526ae00a1f5c3442f768d6108fe680
```

Copy and paste the URL that appears in **your terminal** beginning with `http://localhost` into your web browser.

You should see a file listing of the `ex4` directory. Click on `Visualization.ipynb` and follow the instructions within.
