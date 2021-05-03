/**
@file degnome.c
@page degnome
@author Daniel R. Tabin
@brief Digital Genomes aka Degnomes

This program will be used to simulated Polygenic evoltion of
quantitative traits by using Degnomes as defined above.
*/
#include "cuda_degnome.h"
#include "fitfunc.h"
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
	void Degnome_reorganize(size_t blocksCount, size_t threadsCount, Degnome* q, int pop_size, int chrom_size);
	Degnome* Degnome_cuda_new(int pop_size, int chrom_size);
	// void Degnome_mate(Degnome* child, Degnome* p1, Degnome* p2, curandStateXORWOW_t* state,
	// 				int mutation_rate, int mutation_effect, int crossover_rate, int chrom_size);
	void Degnome_cuda_free(Degnome* q);
}

__global__ void kernel_regorganize(Degnome* q, int pop_size, int chrom_size) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	printf("Index: %u\n", i);

	Degnome* end_of_dengomes = q + pop_size;
	double* ptr_itr = (double*) end_of_dengomes;

	while (index < pop_size) {
		// set the dna array
		q[index].dna_array = ptr_itr + (index * chrom_size);
		// move the pointer
		index += blockDim.x * gridDim.x;
	}
}

void Degnome_reorganize(size_t blocksCount, size_t threadsCount, Degnome* q, int pop_size, int chrom_size) {
	kernel_regorganize<<<blocksCount,threadsCount>>>(q, pop_size, chrom_size);
	
	cudaDeviceSynchronize();
}

Degnome* Degnome_cuda_new(int pop_size, int chrom_size) {
	Degnome* q;

	// calculate the size of a degnmed based on the chromosome length
	int degnome_size = (sizeof(Degnome) + (chrom_size * sizeof(double)));

	// malloc a single chunk of memory for easy MPI transfer
	cudaMallocManaged(&q, (pop_size*degnome_size));

	return q;
}

// // device function
// __device__ void Degnome_mate(Degnome* child, Degnome* p1, Degnome* p2, void* rng_ptr,
// 	int mutation_rate, int mutation_effect, int crossover_rate, int chrom_size) {
// 	// printf("mating\n");

// 	//get rng
// 	curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr;
	
// 	//Cross over
// 	int num_crossover = curand_poisson(state, crossover_rate);
// 	int crossover_locations[num_crossover];
// 	int distance = 0;
// 	int diff;

// 	for (int i = 0; i < num_crossover; i++) {
// 		crossover_locations[i] = (curand_poisson(state) % chrom_size);
// 	}
// 	if (num_crossover > 0) {
// 		int_qsort(crossover_locations, num_crossover);//changed
// 	}

// 	for (int i = 0; i < num_crossover; i++) {
// 		diff = crossover_locations[i] - distance;

// 		if (i % 2 == 0) {
// 			cudaMemcpy(child->dna_array+distance, p1->dna_array+distance, (diff*sizeof(double)), cudaMemcpyDefault);
// 		}
// 		else {
// 			cudaMemcpy(child->dna_array+distance, p2->dna_array+distance, (diff*sizeof(double)), cudaMemcpyDefault);
// 		}
// 		distance = crossover_locations[i];
// 	}

// 	if (num_crossover > 0) {
// 		diff = chrom_size - crossover_locations[num_crossover-1];
// 	}
// 	else {
// 		diff = chrom_size;
// 	}

// 	if (i % 2 == 0) {
// 		cudaMemcpy(child->dna_array+distance, p1->dna_array+distance, (diff*sizeof(double)), cudaMemcpyDefault);
// 	}
// 	else {
// 		cudaMemcpy(child->dna_array+distance, p2->dna_array+distance, (diff*sizeof(double)), cudaMemcpyDefault);
// 	}

// 	child->hat_size = 0;

// 	//mutate
// 	double mutation;
// 	int num_mutations = curand_poisson(state, mutation_rate);
// 	int mutation_location;

// 	for (int i = 0; i < num_mutations; i++) {
// 		mutation_location = (curand_poisson(state) % chrom_size);
// 		mutation = (curand_normal_double(state) * mutation_effect);
// 		child->dna_array[mutation_location] += mutation;
// 	}

// 	//calculate hat_size

// 	for (int i = 0; i < chrom_size; i++) {
// 		child->hat_size += child->dna_array[i];
// 	}

// 	// calculate fitness via cuda

// 	child->fitness = get_fitness(child->hat_size);
// 	//and we are done!
// }

void Degnome_cuda_free(Degnome* q) {
	// no need to free the dna_array as it is part of q
	// I think, we may want to test this
	cudaFree(q);
}