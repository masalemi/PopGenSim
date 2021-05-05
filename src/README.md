# CUDAGenSim

To run CUDAGenSim

1.) Copy the src file into your computing environment
2.) Run the make file with the `make` command
3.) Run CUDAGenSim with the specified number of ranks and devices:
```
sbatch -N 1 --ntasks-per-node=1 --partition=dcs --gres=gpu:1 -t 5 ./slurmSpectrum.sh cudagensim.exe
```
4.) Optional arguments for CUDAGenSim can be adjusted by changing the default in `flagparse.c`