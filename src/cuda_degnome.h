#ifndef DEGNOME
#define DEGNOME

#include <stdlib.h>

//Degnomis Cuda
typedef struct Degnome Degnome;
struct Degnome {
	double* dna_array;
	double hat_size;
	double fitness;

};

extern void Degnome_reorganize(size_t blocksCount, size_t threadsCount, Degnome* q, int pop_size, int chrom_size);
extern Degnome* Degnome_cuda_new(int pop_size, int chrom_size);
// extern void Degnome_mate(Degnome* child, Degnome* p1, Degnome* p2, void* rng_ptr,
// 								int mutation_rate, int mutation_effect,
// 								int crossover_rate, int chrom_size);
extern void Degnome_cuda_free(Degnome* q);

#endif