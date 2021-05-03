#ifndef DEGNOME_GEN_RAND
#define DEGNOME_GEN_RAND

#include <stdlib.h>

/* typedef struct SelectionPair SelectionPair; */
/* struct SelectionPair { */
/* 	double win_m; */
/* 	double win_d; */
/* }; */

/* typedef struct GenomeLocations GenomeLocations; */
/* struct GenomeLocations { */
/* 	int size; */
/* 	int* locations; */
/* }; */

void initializeCudaRandGenerator(size_t blocks,
								 size_t threads_per_block,
								 int rank,
								 int num_ranks,
								 unsigned long long seed);

void freeCudaRandGenerator();

void cudaSelection(int child_pop_size,
				   double* result,
				   double total_hat_size,
				   size_t blocks,
				   size_t threads_per_block);

double* allocatePairArray(int child_pop_size);

double* freePairArray(double* array);

#endif


