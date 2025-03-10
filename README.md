# gpu-workshop-exercises

## MSU HPCC

1. Login to the interactive GPU node here: https://ondemand.hpcc.msu.edu/pun/sys/shell/ssh/dev-amd24-h200

2. Download this repository:
```
$ git clone https://github.com/BenWibking/gpu-workshop-exercises.git
```

3. Load the included Bash profile:
```
$ cd gpu-workshop-exercises
$ . hpcc.profile
```

## OzStar
1. Login to `nt.swin.edu.au`:
```
$ ssh nt.swin.edu.au
```

2. Download this repository:
```
$ git clone https://github.com/BenWibking/gpu-workshop-exercises.git
```

3. Load the included Bash profile
```
$ cd gpu-workshop-exercises
$ . nt.profile
```

4. Test that you are able to launch an interactive GPU job:
```
[bwibking@tooarrana2 gpu-workshop-exercises]$ start_gpu_job
srun: job 64075874 queued and waiting for resources
srun: job 64075874 has been allocated resources

[bwibking@gina13 gpu-workshop-exercises]$ nvidia-smi
Fri Jan 24 06:34:33 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:41:00.0 Off |                    0 |
| N/A   26C    P0             58W /  500W |       1MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

[bwibking@gina13 gpu-workshop-exercises]$ exit
logout
```

Now you are ready to proceed to the exercises.
