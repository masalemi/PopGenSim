#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "degnome.h"
#include "fitfunc.h"

Degnome* parent_gen = NULL;
double* cum_gen = NULL;
// Degnome* temp_gen = NULL;
Degnome* child_gen = NULL;

unsigned int g_pop_size = NULL;
unsigned int g_chrom_size = NULL;
unsigned int g_mutation_rate = NULL;
unsigned int g_mutation_effect = NULL;
unsigned int g_crossover_rate = NULL;
double g_hat_sum = NULL;

extern "C" void cuda_set_device(size_t my_rank) {

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

extern "C" void cuda_set_seed(unsigned long rng_seed) {
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rng, rng_seed);
}

extern "C" Degnome* cuda_calloc_gens(unsigned int pop_size, unsigned int num_ranks, unsigned int chrom_size,
                                        unsigned int mutation_rate, unsigned int mutation_effect,
                                        unsigned int crossover_rate) {

    g_pop_size = pop_size;
    g_chrom_size = chrom_size;
    g_mutation_rate = mutation_rate;
    g_mutation_effect = mutation_effect;
    g_crossover_rate = crossover_rate;

    // Allocate the parent, temp and child generations

    cudaMallocManaged(&parent_gen, (pop_size * sizeof(Degnome)));
    cudaMallocManaged(&cum_gen, (pop_size * sizeof(double)));
    // cudaMallocManaged(&temp_gen, (pop_size * sizeof(Degnome)));
    cudaMallocManaged(&child_gen, ((pop_size / num_ranks) * sizeof(Degnome)));

    for (int i = 0; i < pop_size; i++) {

        cudaMallocManaged(&(parent_gen[i].dna_array), chrom_size * sizeof(double));
        parent_gen[i].hat_size = 0;

        for (int j = 0; j < chrom_size; j++) {
            parent_gen[i].dna_array[j] = (i + j);
            parent_gen[i].hat_size += (i + j);
        }

        // if (i < (pop_size / num_ranks)) {

        //     cudaMallocManaged(&(child_gen[i].dna_array), chrom_size * sizeof(double));
        //     child_gen[i].hat_size = 0;
        // }
    }

    double sum = 0;

    for (int i = 0; i < pop_size; i++) {
        sum += get_fitness(parent_gen[i].hat_size);
        cum_gen[i] = sum;
    }

    g_hat_sum = sum;
}


// Slightly favors 0 index

int binary_search(double value) {

    int l = 0;
    int r = g_pop_size - 1;
    int m = (l + r) / 2

    while (l <= r) {
        if (parent_gen[m] < value) {
            l = m + 1;
        }
        else if (parent_gen[m] > value) {
            r = m - 1;
        }
        else {
            break;
        }
        m = (l + r) / 2;
    }

    return m
}


__global__ void kernel() {

    // Iterate through each index in child generation subset

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            index < rankWidth * rankHeight; index += blockDim.x * gridDim.x) {

        // Generate two random numbers between 0 and sum hat size (what should rng be?)

        double win_m = gsl_rng_uniform(rng);
        win_m *= total_hat_size;
        double win_d = gsl_rng_uniform(rng);
        win_d *= total_hat_size;

        // Use binary search to lookup degnomes of both parents (leave as ints)

        int m_index = binary_search(win_m);
        int d_index = binary_search(win_d);

        Degnome* m = parent_gen[m_index];
        Degnome* d = parent_gen[d_index];

        // Perform crossover

        Degnome* c = NULL;

        // How to handle mate memory?

        Degnome_mate(c, m, d, rng, g_mutation_rate, g_mutation_effect, g_crossover_rate);

        // Write child degnome to current child generation subset index

        child_gen[index] = c;
    }
}

extern "C" void kernel_launch() {

    // Calculate number of blocks / threads

    // Call kernel

    // Sync Devices

    cudaDeviceSynchronize();
}

extern "C" void cuda_update_parents() {

    // Free the current parents

    // for (int i = 0; i < g_pop_size; i++) {
    //     cudaFree(parent_gen[i].dna_array);
    // }

    // cudaFree(parent_gen);

    // Replace current parents with temp

    // parent_gen = temp_gen;

    // Allocate new buffer

    // cudaMallocManaged(&temp_gen, (pop_size * sizeof(Degnome)));

    // Create new cumulative generation

    double sum = 0;

    for (int i = 0; i < pop_size; i++) {
        sum += get_fitness(parent_gen[i].hat_size);
        cum_gen[i] = sum;
    }

    g_hat_sum = sum;
}


extern "C" void cuda_print_parents(unsigned int num_gens) {

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

extern "C" void cuda_free_gens() {

    for (int i = 0; i < g_pop_size; i++) {
        cudaFree(parent_gen[i].dna_array);
    }

    cudaFree(parent_gen);
    cudaFree(cum_gen);
    // cudaFree(temp_gen);
    cudaFree(child_gen);
}