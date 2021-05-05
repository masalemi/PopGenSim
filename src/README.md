# CUDAGenSim

To run CUDAGenSim

1.) Run the make file with the `make` command
2.) Run CUDAGenSim with the specified number of ranks and devices:
```
sbatch -N 1 --ntasks-per-node=1 --partition=dcs --gres=gpu:1 -t 5 ./slurmSpectrum.sh cudagensim.exe
```
3.) Optional arguments for CUDAGenSim can be adjusted by changing the default in `flagparse.c`