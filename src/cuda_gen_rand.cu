/**
 * This module is responsible for generating mate pairs and crossovers
 * for use by the main kernel. Note that the values are generated in
 * *device* memory (to avoid copy overhead). The values are not intended
 * for use in host memory, so cudaManagedMalloc is not needed.
 *
 * Make sure to call initializeRandGenerator bofore generating any numbers.
 * Some functions will return a curandStatus_t. This can be used for extra
 * error checking if necessary.
 * 
 * @author: Owen Xie
 */

#include <cuda.h>
#include <cuda_kernel.h> // The kernel-level API
#include "cuda_gen_ran.h"

__device__ curandState* cuda_rand_state;

__global__ void setupKernel(curandState* state,
							int rank,
							int num_ranks,
							unsigned long long seed) {
	// Only need as many states as there are threads - can share among popuation
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	// Ensure generators across all ranks start on different numbers
	unsigned long long sequence = id * rank * num_ranks;
	curand_init(seed, sequence, 0, &state[id]);
}

__global__ void uniformKernel(curandState* state,
							  double* result,
							  int pair_array_size,
							  double total_hat_size) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	int stride = gridDim.x * blockDim.x; // Total threads in grid
	curandState local_state = state[id]; // Copy to local memory for efficiency
	for (int index = id; index < pair_array_size; index += stride) {
		double choice = curand_uniform_double(local_state);
		choice *= total_hat_size;
		result[index] = choice;
	}
	state[id] = localState; // Copy state back to global memory
}

__global__ void poissonKernel(curandState* state,
							  double* )

void initializeCudaRandGenerator(size_t blocks,
								 size_t threads_per_block,
								 int rank,
								 int num_ranks,
								 unsigned long long seed) {
	int total_threads = blocks * threads_per_block;
	cudaMalloc(&cuda_rand_state, total_threads * sizeof(curandState));
	setupKernel<<<blocks, threads_per_block>>>(cuda_rand_state, rank, num_ranks, seed);
}

void freeCudaRandGenerator() {
	cudaFree(cuda_rand_state);
}

void cudaSelection(int child_pop_size,
				   double* result,
				   double total_hat_size,
				   size_t blocks,
				   size_t threads_per_block) {
	size_t generate_count = 2 * child_pop_size;
	uniformKernel<<<blocks, threads_per_block>>>(cuda_rand_state, result, generate_count, total_hat_size);
	cudaDeviceSynchronize();
}

// Allocates an array with 2 * size doubles.
double* allocatePairArray(int child_pop_size) {
	int length = 2 * child_pop_size;
	cudaMalloc(&array, length * sizeof(double));
	return array;
}

double* freePairArray(double* array) {
	cudaFree(array);
}