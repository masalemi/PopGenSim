#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cuda_degnome.h"
#include "fitfunc.h"

// Degnome* parent_gen = NULL;
// double* cum_siz_arr = NULL;
// // Degnome* temp_gen = NULL;
// Degnome* child_gen = NULL;

// unsigned int g_pop_size = NULL;
// unsigned int g_chrom_size = NULL;
// unsigned int g_mutation_rate = NULL;
// unsigned int g_mutation_effect = NULL;
// unsigned int g_crossover_rate = NULL;
// double g_hat_sum = NULL;

extern "C" {
    void cuda_set_device(size_t my_rank);
    void* cuda_set_seed(size_t blocksCount, size_t threadsCount, unsigned long rng_seed, unsigned int pop_size);
    void kernel_launch(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
                        int child_pop_size, double total_hat_size, double* cum_siz_arr,
                        void* rng_ptr, size_t blocksCount, size_t threadsCount);
    // cuda_update_parents();
    cuda_print_parents(unsigned int num_gens);
    // void cuda_free_gens();
    void cuda_free_rng(void* rng_ptr);
}

void cuda_set_device(size_t my_rank) {

    // Set the cuda device baesd on rank index

    int cudaDeviceCount;
    cudaError_t cE;

    if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess) {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }
    if ((cE = cudaSetDevice(my_rank % cudaDeviceCount)) != cudaSuccess) {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }
}

// should be long long
// THIS FIXES THE ISSUE OF MULTITHREAD DETERMENISM <3 <3 <3
void* cuda_set_seed(size_t blocksCount, size_t threadsCount, int my_rank, unsigned long rng_seed, unsigned int pop_size) {
    curandStateXORWOW_t* state;
    cudaMallocManaged(&state, (pop_size * sizeof(curandStateXORWOW_t)));

    kernel_setup_rng<<<blocksCount,threadsCount>>>(state, my_rank, rng_seed, pop_size);

    cudaDeviceSynchronize();

    void* rng_ptr = (void*) state;
    return rng_ptr; 

}

__global__ void kernel_setup_rng(curandStateXORWOW_t* state, int my_rank, unsigned long rng_seed, unsigned int pop_size) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    while (index < pop_size) {
        // Not sure if this is the best way of doing things;
        // this is based on examples from online
        // possibly we should add index to the seed
        curand_init(rng_seed, (index + (my_rank*pop_size)), 0, &state[index]);

        index += blockDim.x * gridDim.x;
    }
}

// Degnome* cuda_calloc_gens(unsigned int pop_size, unsigned int num_ranks, unsigned int chrom_size,
//                                         unsigned int mutation_rate, unsigned int mutation_effect,
//                                         unsigned int crossover_rate) {

//     g_pop_size = pop_size;
//     g_chrom_size = chrom_size;
//     g_mutation_rate = mutation_rate;
//     g_mutation_effect = mutation_effect;
//     g_crossover_rate = crossover_rate;

//     // Allocate the parent, temp and child generations

//     cudaMallocManaged(&parent_gen, (pop_size * sizeof(Degnome)));
//     cudaMallocManaged(&cum_siz_arr, (pop_size * sizeof(double)));
//     // cudaMallocManaged(&temp_gen, (pop_size * sizeof(Degnome)));
//     cudaMallocManaged(&child_gen, ((pop_size / num_ranks) * sizeof(Degnome)));

//     for (int i = 0; i < pop_size; i++) {

//         cudaMallocManaged(&(parent_gen[i].dna_array), chrom_size * sizeof(double));
//         parent_gen[i].hat_size = 0;

//         for (int j = 0; j < chrom_size; j++) {
//             parent_gen[i].dna_array[j] = (i + j);
//             parent_gen[i].hat_size += (i + j);
//         }

//         // if (i < (pop_size / num_ranks)) {

//         //     cudaMallocManaged(&(child_gen[i].dna_array), chrom_size * sizeof(double));
//         //     child_gen[i].hat_size = 0;
//         // }
//     }

//     double sum = 0;

//     for (int i = 0; i < pop_size; i++) {
//         sum += get_fitness(parent_gen[i].hat_size);
//         cum_siz_arr[i] = sum;
//     }

//     g_hat_sum = sum;
// }


// Slightly favors 0 index

int binary_search(double value, int pop_size, double* search_arr) {

    int l = 0;
    int r = pop_size - 1;
    int m = (l + r) / 2

    while (l <= r) {
        if (search_arr[m] < value) {
            l = m + 1;
        }
        else if (search_arr[m] > value) {
            r = m - 1;
        }
        else {
            break;
        }
        m = (l + r) / 2;
    }

    return m
}


__global__ void kernel_mate(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
                        int child_pop_size, double total_hat_size, double* cum_siz_arr,
                        curandStateXORWOW_t* state) {

    // Iterate through each index in child generation subset

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    while (index < child_pop_size) {

        // get our random number generator

        curandStateXORWOW_t* rng = &state[index];

        // Generate two random numbers between 0 and sum hat size

        double win_m = curand_uniform_double(rng);
        win_m *= total_hat_size;
        double win_d = curand_uniform_double(rng);
        win_d *= total_hat_size;

        // Use binary search to lookup degnomes of both parents (leave as ints)

        int m_index = binary_search(win_m, parent_pop_size, cum_siz_arr);
        int d_index = binary_search(win_d, parent_pop_size, cum_siz_arr);

        // get the parents
        Degnome* m = parent_gen[m_index];
        Degnome* d = parent_gen[d_index];

        // child is just our index

        Degnome* c = child_gen[index];

        // mate degnomes

        Degnome_mate(c, m, d, rng, g_mutation_rate, g_mutation_effect, g_crossover_rate);
    }
}

void kernel_launch(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
                    int child_pop_size, double total_hat_size, double* cum_siz_arr,
                    void* rng_ptr, size_t blocksCount, size_t threadsCount) {

    // get rng
    curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr
    // Call kernel
    kernel_mate<<<blocksCount,threadsCount>>>(parent_arr, child_arr, parent_pop_size, child_pop_size, total_hat_size, cum_siz_arr, state);

    // Sync Devices

    cudaDeviceSynchronize();
}

// Z


void cuda_print_parents(unsigned int num_gens) {

    // Print info for parent generation

    printf("Generation %u:\n", num_gens);
    for (int i = 0; i < g_pop_size; i++) {
        printf("Degnome %u\n", i);
        for (int j = 0; j < g_chrom_size; j++) {
            printf("%lf\t", parent_gen[i].dna_array[j]);
        }
        printf("\nTOTAL HAT SIZE: %lg\n\n", parent_gen[i].hat_size);
    }
}

// void cuda_free_gens() {

//     for (int i = 0; i < g_pop_size; i++) {
//         cudaFree(parent_gen[i].dna_array);
//     }

//     cudaFree(parent_gen);
//     cudaFree(cum_siz_arr);
//     // cudaFree(temp_gen);
//     cudaFree(child_gen);
// }

void* cuda_free_rng(void* rng_ptr) {
    curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr;
    cudaFree(state);
}