/**
 * This module is responsible for generating mate pairs and crossovers
 * for use by the main kernel. Note that the values are generated in
 * *device* memory (to avoid copy overhead). The values are not intended
 * for use in host memory, so cudaManagedMalloc is not needed.
 *
 * Make sure to call initializeRandGenerator before generating any numbers.
 *
 * @author: Owen Xie
 */
#include <cuda.h>
#include <stdbool.h>
#include <curand_kernel.h> // The kernel-level API
#include "cuda_gen_rand.h"

extern "C" {
    void initializeCudaRandGenerator(size_t blocks,
                                     size_t threads_per_block,
                                     unsigned int rank,
                                     unsigned int num_ranks,
                                     unsigned long long seed);
    void freeCudaRandGenerator();
    void cudaSelection(unsigned int child_pop_size,
                       double* result,
                       double total_hat_size,
                       size_t blocks,
                       size_t threads_per_block);
    void cudaMarking(unsigned int pop_size,
                     unsigned int chrom_size,
                     bool* result,
                     size_t blocks,
                     size_t threads_per_block,
                     double probability);

    bool* allocateMarkArray(unsigned int pop_size, unsigned int chrom_size);
    bool* getDegnomeMarks(bool* array, unsigned int chrom_size, unsigned int i);
    void freeMarkArray(bool* array);
    double* allocatePairArray(unsigned int child_pop_size);
    void freePairArray(double* array);
}

curandState* cuda_rand_state;

__global__ void setupKernel(curandState* state,
                            unsigned int rank,
                            unsigned int num_ranks,
                            unsigned long long seed) {
    // Only need as many states as there are threads - can share among population
    unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    // Ensure generators across all ranks start on different numbers
    unsigned long long sequence = id * rank * num_ranks;
    curand_init(seed, sequence, 0, &state[id]);
}

__global__ void uniformKernel(curandState* state,
                              double* result,
                              unsigned int pair_array_size,
                              double total_hat_size) {
    unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int stride = gridDim.x * blockDim.x; // Total threads in grid
    curandState local_state = state[id]; // Copy to local memory for efficiency
    for (unsigned int index = id; index < pair_array_size; index += stride) {
        double choice = curand_uniform_double(&local_state);
        choice *= total_hat_size;
        result[index] = choice;
    }
    state[id] = local_state; // Copy state back to global memory
}

__global__ void repeatedBernoulli(curandState* state,
                                  bool* result,
                                  unsigned int arr_size,
                                  double probability) {
    unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int stride = gridDim.x * blockDim.x;
    curandState  local_state = state[id];
    for (unsigned int index = id; index < arr_size; index += stride) {
        double roll = curand_uniform_double(&local_state);
        if (roll < probability) result[index] = true;
    }
    state[id] = local_state;
}

void initializeCudaRandGenerator(size_t blocks,
                                 size_t threads_per_block,
                                 unsigned int rank,
                                 unsigned int num_ranks,
                                 unsigned long long seed) {
    unsigned int total_threads = blocks * threads_per_block;
    cudaMalloc(&cuda_rand_state, total_threads * sizeof(curandState));
    setupKernel<<<blocks, threads_per_block>>>(cuda_rand_state, rank, num_ranks, seed);
}

void freeCudaRandGenerator() {
    cudaFree(cuda_rand_state);
}

void cudaSelection(unsigned int child_pop_size,
                   double* result,
                   double total_hat_size,
                   size_t blocks,
                   size_t threads_per_block) {
    unsigned int generate_count = 2 * child_pop_size;
    uniformKernel<<<blocks, threads_per_block>>>(cuda_rand_state, result, generate_count, total_hat_size);
    cudaDeviceSynchronize();
}

void cudaMarking(unsigned int pop_size,
                 unsigned int chrom_size,
                 bool* result,
                 size_t blocks,
                 size_t threads_per_block,
                 double probability) {
    unsigned int generate_count = pop_size * chrom_size;
    repeatedBernoulli<<<blocks, threads_per_block>>>(cuda_rand_state, result, generate_count, probability);
    cudaDeviceSynchronize();
}


bool* allocateMarkArray(unsigned int pop_size, unsigned int chrom_size) {
    unsigned int generate_count = pop_size * chrom_size;
    bool* array;
    cudaMalloc(&array, generate_count * sizeof(double));
    return array;
}

bool* getDegnomeMarks(bool* array, unsigned int chrom_size, unsigned int i) {
    return array + (chrom_size * i);
}

void freeMarkArray(bool* array) {
    cudaFree(array);
}

// Allocates an array with 2 * size doubles.
double* allocatePairArray(unsigned int child_pop_size) {
    unsigned int length = 2 * child_pop_size;
    double* array;
    cudaMalloc(&array, length * sizeof(double));
    return array;
}

void freePairArray(double* array) {
    cudaFree(array);
}
