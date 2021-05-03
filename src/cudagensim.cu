#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
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
	void* cuda_set_seed(size_t blocksCount, size_t threadsCount, int my_rank, unsigned long rng_seed, unsigned int pop_size);
	void kernel_launch(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
					int child_pop_size, double total_hat_size, double* cum_siz_arr,
					double mutation_rate, double mutation_effect, double crossover_rate,
					int chrom_size, int** cros_loc_arr, void* rng_ptr, size_t blocksCount,
					size_t threadsCount);
	// cuda_update_parents();
	void cuda_print_parents(unsigned int num_gens, Degnome* parent_gen, int pop_size, int chrom_size);
	// void cuda_free_gens();
	void cuda_free_rng(void* rng_ptr);
	int** cuda_malloc_cross_loc_arr(int child_pop_size, int chrom_size);
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
__global__ void kernel_setup_rng (curandStateXORWOW_t* state, int my_rank, unsigned long rng_seed, unsigned int pop_size) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	while (index < pop_size) {
		// Not sure if this is the best way of doing things;
		// this is based on examples from online
		// possibly we should add index to the seed
		curand_init(rng_seed, (index + (my_rank*pop_size)), 0, &state[index]);

		index += blockDim.x * gridDim.x;
	}
}

// should be long long
// THIS FIXES THE ISSUE OF MULTITHREAD DETERMENISM <3 <3 <3
void* cuda_set_seed(size_t blocksCount, size_t threadsCount, int my_rank, unsigned long rng_seed, unsigned int pop_size) {
	curandStateXORWOW_t* state = NULL;
	cudaMallocManaged(&state, (pop_size * sizeof(curandStateXORWOW_t)));

	kernel_setup_rng<<<blocksCount,threadsCount>>>(state, my_rank, rng_seed, pop_size);

	cudaDeviceSynchronize();

	void* rng_ptr = (void*) state;
	return rng_ptr; 

}

int** cuda_malloc_cross_loc_arr(int child_pop_size, int chrom_size) {
	int** cros_loc_arr = NULL;
	cudaMallocManaged(&cros_loc_arr, (child_pop_size * sizeof(int*)));

	for (int i = 0; i < child_pop_size; i++) {
		cudaMallocManaged((cros_loc_arr + i), (chrom_size * sizeof(int)));
	}

	return cros_loc_arr;
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

__device__ void binary_search(double value, int pop_size, double* search_arr, int* result) {

	int l = 0;
	int r = pop_size - 1;
	int m = (l + r) / 2;

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

	*result = m;
}

__device__ void int_qsort(int* arr, int arr_len) {
	if (arr_len <= 1) {
		return;
	}

	int pivot = arr[0];
	int swaps = 0;
	int temp;

	for (int i = 0; i < arr_len;i++){
		if (arr[i] < pivot){
			temp = arr[swaps];
			arr[swaps] = arr[i];
			arr[i] = temp;

			swaps++;
		}
	}

	int_qsort(arr, (swaps-1));
	int_qsort((arr + (swaps+1)), (arr_len - (swaps+1)));

}

// device function
__device__ void Degnome_mate(Degnome* child, Degnome* p1, Degnome* p2, void* rng_ptr,
							int mutation_rate, int mutation_effect, int crossover_rate,
							int* crossover_locations, int chrom_size) {
	// printf("mating\n");

	//get rng
	curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr;
	
	//Cross over
	int num_crossover = curand_poisson(state, crossover_rate);

	// prevent overflow
	while (num_crossover >= chrom_size) {
		num_crossover = curand_poisson(state, crossover_rate);
	}

	int distance = 0;
	int diff;

	// int* crossover_locations = (int*) malloc(num_crossover*sizeof(int));

	// only init as far as num_crossover
	for (int i = 0; i < num_crossover; i++) {
		crossover_locations[i] = (curand(state) % chrom_size);
	}

	// only sort the num_crossover part
	if (num_crossover > 0) {
		int_qsort(crossover_locations, num_crossover);
	}

	for (int i = 0; i < num_crossover; i++) {
		diff = crossover_locations[i] - distance;

		if (i % 2 == 0) {
			memcpy(child->dna_array+distance, p1->dna_array+distance, (diff*sizeof(double)));
		}
		else {
			memcpy(child->dna_array+distance, p2->dna_array+distance, (diff*sizeof(double)));
		}
		distance = crossover_locations[i];
	}

	if (num_crossover > 0) {
		diff = chrom_size - crossover_locations[num_crossover-1];
	}
	else {
		diff = chrom_size;
	}

	if (num_crossover % 2 == 0) {
		memcpy(child->dna_array+distance, p1->dna_array+distance, (diff*sizeof(double)));
	}
	else {
		memcpy(child->dna_array+distance, p2->dna_array+distance, (diff*sizeof(double)));
	}

	child->hat_size = 0;

	//mutate
	double mutation;
	int num_mutations = curand_poisson(state, mutation_rate);
	int mutation_location;

	for (int i = 0; i < num_mutations; i++) {
		mutation_location = (curand(state) % chrom_size);
		mutation = (curand_normal_double(state) * mutation_effect);
		child->dna_array[mutation_location] += mutation;
	}

	//calculate hat_size

	for (int i = 0; i < chrom_size; i++) {
		child->hat_size += child->dna_array[i];
	}

	// calculate fitness via cuda

	// free(crossover_locations);

	// child->fitness = get_fitness(child->hat_size);
	//and we are done!
}

__global__ void kernel_select_and_mate (Degnome* parent_gen, Degnome* child_gen, int parent_pop_size,
										int child_pop_size, double total_hat_size, double* cum_siz_arr,
										double mutation_rate, double mutation_effect, double crossover_rate,
										int chrom_size, int** cros_loc_arr, curandStateXORWOW_t* state) {

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

		int m_index = 0;
		int d_index = 0;

		binary_search(win_m, parent_pop_size, cum_siz_arr, &m_index);
		binary_search(win_d, parent_pop_size, cum_siz_arr, &d_index);

		// get the parents
		Degnome* m = parent_gen + m_index;
		Degnome* d = parent_gen + d_index;

		// child is just our index

		Degnome* c = child_gen + index;

		// mate degnomes

		Degnome_mate(c, m, d, rng, mutation_rate, mutation_effect, crossover_rate, cros_loc_arr[index], chrom_size);
	}
}

void kernel_launch(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
					int child_pop_size, double total_hat_size, double* cum_siz_arr,
					double mutation_rate, double mutation_effect, double crossover_rate,
					int chrom_size, int** cros_loc_arr, void* rng_ptr, size_t blocksCount,
					size_t threadsCount) {

	// get rng
	curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr;
	// Call kernel
	kernel_select_and_mate<<<blocksCount,threadsCount>>>(parent_arr, child_arr, parent_pop_size,
		child_pop_size, total_hat_size, cum_siz_arr, mutation_rate, mutation_effect,
		crossover_rate, chrom_size, cros_loc_arr, state);

	// Sync Devices

	cudaDeviceSynchronize();
}

// Z


void cuda_print_parents(unsigned int num_gens, Degnome* parent_gen, int pop_size, int chrom_size) {

	// Print info for parent generation

	printf("Generation %u:\n", num_gens);
	for (int i = 0; i < pop_size; i++) {
		printf("Degnome %u\n", i);
		for (int j = 0; j < chrom_size; j++) {
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

void cuda_free_rng(void* rng_ptr) {
	curandStateXORWOW_t* state = (curandStateXORWOW_t*) rng_ptr;
	cudaFree(state);
}