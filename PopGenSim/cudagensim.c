#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "cuda_gen_rand.h"
#include "cuda_degnome.h"

extern void cudaSetup(int rank);
extern void launchKernel(double* parent_gen_chrom, unsigned int parent_pop_size,
                         double* child_gen_chrom, unsigned int child_pop_size,
                         double* pair_array,
                         double* cumulative_array,
                         bool* crossovers, bool* mutations,
                         size_t blocks, size_t threads_per_block,
                         unsigned int chrom_size);
extern double* allocateDegnomes(unsigned int pop_size, unsigned int chrom_size);
extern double* getDegnome(double* array, unsigned int chrom_size, unsigned int i);
extern void freeDegnomes(double* array);
extern double* allocateCudaArray(unsigned int size);
extern void freeCudaArray(double* array);
extern void initializeCudaRandGenerator(size_t blocks,
                                 size_t threads_per_block,
                                 unsigned int rank,
                                 unsigned int num_ranks,
                                 unsigned long long seed);
extern void freeCudaRandGenerator();
extern void cudaSelection(unsigned int child_pop_size,
                   double* result,
                   double total_hat_size,
                   size_t blocks,
                   size_t threads_per_block);
extern void cudaMarking(unsigned int pop_size,
                 unsigned int chrom_size,
                 bool* result,
                 size_t blocks,
                 size_t threads_per_block,
                 double probability);

extern bool* allocateMarkArray(unsigned int pop_size, unsigned int chrom_size);
extern bool* getDegnomeMarks(bool* array, unsigned int chrom_size, unsigned int i);
extern void freeMarkArray(bool* array);
extern double* allocatePairArray(unsigned int child_pop_size);
extern void freePairArray(double* array);

int main() {
    unsigned long rng_seed = 1234;
    unsigned int pop_size = 10;
    unsigned int num_gens = 1000;
    unsigned int mutation_rate = 1;
    unsigned int mutation_effect = 2;
    unsigned int crossover_rate = 2;

    unsigned int chrom_size = 10;
    double mutation_prob = ((double) mutation_effect) / chrom_size;
    double crossover_prob = ((double) crossover_rate) / chrom_size;

    unsigned long cur_time = (unsigned long) MPI_Wtime();

    // Cuda specific stuff
    size_t threads_per_block = 1;
    size_t blocks = 1;

    int rank, num_ranks;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    cudaSetup(rank);
    initializeCudaRandGenerator(blocks, threads_per_block, rank, num_ranks, rng_seed);

    // Generation data
    unsigned int child_pop_size = pop_size / num_ranks;
    double* parent_gen_chrom = allocateDegnomes(pop_size, chrom_size);
    double* child_gen_chrom = allocateDegnomes(child_pop_size, chrom_size);

    double* pair_array = allocatePairArray(child_pop_size);
    bool* crossovers = allocateMarkArray(child_pop_size, chrom_size);
    bool* mutations = allocateMarkArray(child_pop_size, chrom_size);

    // Statistics allocation
    // For choosing parents
    // Stats are calculated for children, shared as parents.
    double* cumulative_array = allocateCudaArray(pop_size);
    double* hat_size = allocateCudaArray(child_pop_size);
    double* fitness = allocateCudaArray(child_pop_size);

    // Initialize degnomes
    for (unsigned int i = 0; i < child_pop_size; i++) {
        hat_size[i] = 0;

//        double* offset = getDegnome(child_gen_chrom, chrom_size, i);
        unsigned int offset = chrom_size * i;
        for (unsigned int j = 0; j < chrom_size; j++) {
            unsigned int dna = i + j + rank;
            *(child_gen_chrom + offset + j) = dna;
            hat_size[i] += dna;
        }
        fitness[i] = hat_size[i]; // Default to linaer returns
    }

    unsigned int send_degnome_count = child_pop_size * chrom_size;
    unsigned int recv_degnome_count = pop_size * chrom_size;
    unsigned int send_fitness_count = child_pop_size;
    unsigned int recv_fitness_count = pop_size;

    double* parent_fitness = allocateCudaArray(pop_size);

    for (unsigned int tick = 0; tick < num_gens; tick++) {
        MPI_Allgather(child_gen_chrom, send_degnome_count, MPI_DOUBLE,
                      parent_gen_chrom, recv_degnome_count, MPI_DOUBLE, MPI_COMM_WORLD);

        MPI_Allgather(fitness, send_fitness_count, MPI_DOUBLE,
                      parent_fitness, recv_fitness_count, MPI_DOUBLE, MPI_COMM_WORLD);

        double total_hat_size = fitness[0];
        cumulative_array[0] = total_hat_size;
        for (unsigned int i = 1; i < pop_size; i++) {
            // parent fitness already sent
            total_hat_size += fitness[i];
            cumulative_array[i] = cumulative_array[i - 1] + fitness[i];
        }

        cudaSelection(child_pop_size, pair_array, total_hat_size, blocks, threads_per_block);
        cudaMarking(child_pop_size, chrom_size, crossovers, blocks, threads_per_block, crossover_prob);
        cudaMarking(child_pop_size, chrom_size, mutations, blocks, threads_per_block, mutation_prob);

        launchKernel(parent_gen_chrom, pop_size, child_gen_chrom, child_pop_size,
                     pair_array, cumulative_array,
                     crossovers, mutations,
                     blocks, threads_per_block,
                     chrom_size);

        for (unsigned int i = 0; i < child_pop_size; i++) {
            hat_size[i] = 0;

            unsigned int offset = chrom_size * i;
            for (unsigned int j = 0; j < chrom_size; j++) {
                hat_size[i] += *(child_gen_chrom + offset + j);
            }
            fitness[i] = hat_size[i]; // Default to linaer returns
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    freeCudaArray(parent_fitness);
    freeCudaArray(cumulative_array);
    freeMarkArray(crossovers);
    freeMarkArray(mutations);
    freePairArray(pair_array);
    freeDegnomes(parent_gen_chrom);
    freeDegnomes(child_gen_chrom);

    return 0;
}


