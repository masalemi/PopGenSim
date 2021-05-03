#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include "fitfunc.h"
#include "flagparse.h"
#include "cuda_degnome.h"

// Declare all extern functions

extern void cuda_set_device(size_t my_rank);
extern void* cuda_set_seed(size_t blocksCount, size_t threadsCount, int my_rank, unsigned long rng_seed, unsigned int pop_size);
extern void cuda_calloc_gens(unsigned int pop_size, unsigned int num_ranks, unsigned int chrom_size);
extern void kernel_launch(Degnome* parent_arr, Degnome* child_arr, int parent_pop_size,
					int child_pop_size, double total_hat_size, double* cum_siz_arr,
					double mutation_rate, double mutation_effect, double crossover_rate,
					int chrom_size, int** cros_loc_arr, void* rng_ptr, size_t blocksCount,
					size_t threadsCount);
extern void cuda_update_parents();
extern void cuda_print_parents(unsigned int num_gens, Degnome* parent_gen, int pop_size, int chrom_size);
extern void cuda_free_gens();
extern void cuda_free_rng(void* rng);
extern int** cuda_malloc_cross_loc_arr(int child_pop_size, int chrom_size);
extern double* cuda_make_cuda_array(int pop_size);

extern void Degnome_reorganize(size_t blocksCount, size_t threadsCount, Degnome* q, int pop_size, int chrom_size);
extern Degnome* Degnome_cuda_new(int pop_size, int chrom_size);
// extern void Degnome_mate(Degnome* child, Degnome* p1, Degnome* p2, void* rng_ptr,
// 								int mutation_rate, int mutation_effect,
// 								int crossover_rate, int chrom_size);
extern void Degnome_cuda_free(Degnome* q);


// Usage information

void usage(void);
void help_menu(void);

const char* usageMsg =
	"Usage: polygensim [-h] [-c chromosome_length] [-e mutation_effect]\n"
	"\t\t  [-g num_generations] [-m mutation_rate]\n"
	"\t\t  [-o crossover_rate] [-p population_size]\n"
	"\t\t  [-t num_threads] [--seed rngseed]\n"
	"\t\t  [--target hat_height target]\n"
	"\t\t  [--sqrt | --linear | --close | --ceiling | --log]\n";

const char* helpMsg =
	"OPTIONS\n"
	"\t -c chromosome_length\n"
	"\t\t Set chromosome length for the current simulation.\n"
	"\t\t Default chromosome length is 10.\n\n"
	"\t -e mutation_effect\n"
	"\t\t Set how much a mutation will effect a gene on average.\n"
	"\t\t Default mutation effect is 2.\n\n"
	"\t -g num_generations\n"
	"\t\t Set how many generations this simulation will run for.\n"
	"\t\t Default number of generations is 1000.\n\n"
	"\t -h\t Display this help menu.\n\n"
	"\t -m mutation_rate\n"
	"\t\t Set the mutation rate for the current simulation.\n"
	"\t\t Default mutation rate is 1.\n\n"
	"\t -o crossover_rate\n"
	"\t\t Set the crossover rate for the current simulation.\n"
	"\t\t Default crossover rate is 2.\n\n"
	"\t -p population_size\n"
	"\t\t Set the population size for the current simulation.\n"
	"\t\t Default population size is 10.\n"
	"\t -t num_threads\n"
	"\t\t Select the number of threads to be used in the current run.\n"
	"\t\t Default is 0 (which will result in 3/4 of cores being used).\n"
	"\t\t Must be 1 if a seed is used in order to prevent race conditions.\n\n"
	"\t --seed rngseed\n"
	"\t\t Select the seed used by the RNG in the current run.\n"
	"\t\t Default seed is 0 (which will result in a random seed).\n\n"
	"\t --target hat_height target\n"
	"\t\t Sets the ideal hat height for the current simulation\n"
	"\t\t Used for fitness functions that have an \"ideal\" value.\n\n"
	"\t --sqrt\t\t fitness will be sqrt(hat_height)\n\n"
	"\t --linear\t fitness will be hat_height\n\n"
	"\t --close\t fitness will be (target - abs(target - hat_height))\n\n"
	"\t --ceiling\t fitness will quickly level off after passing target\n\n";

void usage(void) {
	fputs(usageMsg, stderr);
	exit(EXIT_FAILURE);
}

void help_menu(void) {
	fputs(helpMsg, stderr);
	exit(EXIT_FAILURE);
}

unsigned long rng_seed = 0;

int pop_size;
int num_gens;
int mutation_rate;
int mutation_effect;
int crossover_rate;

int main(int argc, char const *argv[]) {

	setvbuf(stdout, NULL, _IONBF, 0);

	// Read in arguments

	int * flags = NULL;

	if (parse_flags(argc, argv, 1, &flags) == -1) {
		free(flags);
		usage();
	}

	if (flags[2] == 1) {
		free(flags);
		help_menu();
	}

	int chrom_size = flags[6];
	double mutation_effect = flags[7];
	int num_gens = flags[8];
	double mutation_rate = flags[9];
	double crossover_rate = flags[10];
	int pop_size = flags[11];

	int num_threads = flags[12];

	if (flags[13] == 0) {
		set_function("linear");
	}
	else if (flags[13] == 1) {
		set_function("sqrt");
	}
	else if (flags[13] == 2) {
		set_function("close");
	}
	else if (flags[13] == 3) {
		set_function("ceiling");
	}
	else if (flags[13] == 4) {
		set_function("log");
	}
	double target_num = flags[14];

	if (flags[15] <= 0) {
		unsigned long currtime = (unsigned long) MPI_Wtime();         // time
		unsigned long pid = (unsigned long) getpid();  // process id
		rng_seed = currtime ^ pid;                      // random seed
	}
	else {
		rng_seed = flags[15];
	}

	// make this command line args
	size_t threadsCount = 1;
	size_t blocksCount = 1;

	free(flags);

	// Set up MPI stuff (init and rank number)

	int my_rank, num_ranks;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	// Set CUDA Device Based on MPI rank

	cuda_set_device(my_rank);

	// calculate size of children generation
	int child_pop_size = pop_size / num_ranks;

	// Set random seed

	void* rng_ptr = cuda_set_seed(blocksCount, threadsCount, my_rank, rng_seed, child_pop_size);

	// Initialize local information and cuda calloc the memory we need (See lines 192 - 204)

	Degnome* parent_gen = Degnome_cuda_new(pop_size, chrom_size);
	Degnome* child_gen = Degnome_cuda_new(child_pop_size, chrom_size);

	int** cros_loc_arr = cuda_malloc_cross_loc_arr(child_pop_size, chrom_size);

	double fit;
	double total_hat_size;
	double* cum_siz_arr = cuda_make_cuda_array(pop_size);

	// Degnome_reorganize(Degnome* parent_gen, int pop_size, int chrom_size);
	Degnome_reorganize(blocksCount, threadsCount, child_gen, child_pop_size, chrom_size);

	unsigned long long int print_me = 0;
	printf("%llu\n", print_me);

	print_me = (unsigned long long int) child_gen;
	printf("%llu\n", print_me);


	for (int i = 0; i < child_pop_size; i++) {
		printf("CHILD %u\n", i);
		print_me = (unsigned long long int) ((void*) child_gen + i);
		printf("%llu\n", print_me);
		print_me = (unsigned long long int) ((void*) child_gen[i].dna_array);
		printf("%llu\n", print_me);
	}	

	printf("all done\n");


	//initialize degnomes
	for (int i = 0; i < child_pop_size; i++) {
		child_gen[i].hat_size = 0;

		for (int j = 0; j < chrom_size; j++) {
			child_gen[i].dna_array[j] = (i+j+my_rank);	//children isn't initiilized
			child_gen[i].hat_size += (i+j+my_rank);
		}
		child_gen[i].fitness = get_fitness(child_gen[i].hat_size);
	}

	// get sizes of bytes to send
	int degnome_size = (sizeof(Degnome) + (chrom_size * sizeof(double)));

	int send_bytes = child_pop_size*degnome_size;
	int recv_bytes = pop_size*degnome_size;

	if (my_rank == 0) {
		cuda_print_parents(999, child_gen, child_pop_size, chrom_size);
	}

	for (int i = 0; i < num_gens; i++) {

		// Collect info from all other ranks to make a complete generation
		MPI_Allgather(child_gen, send_bytes, MPI_BYTE, parent_gen, recv_bytes, MPI_BYTE, MPI_COMM_WORLD);

		if (my_rank == 0) {
			cuda_print_parents(i, parent_gen, pop_size, chrom_size);
		}

		// get the pointers right
		Degnome_reorganize(blocksCount, threadsCount, parent_gen, pop_size, chrom_size);

		// make cum_array
		for (int j = 1; j < pop_size; j++) {
			parent_gen[j].fitness = get_fitness(parent_gen[j].hat_size);
			fit = parent_gen[j].fitness;

			total_hat_size += fit;
			cum_siz_arr[j] = (cum_siz_arr[j-1] + fit);
		}

		// make child generation 
		kernel_launch(parent_gen, child_gen, pop_size, child_pop_size, total_hat_size,
						cum_siz_arr, mutation_rate, mutation_effect, crossover_rate,
						chrom_size, cros_loc_arr, rng_ptr, blocksCount, threadsCount);

		// TODO: make more parrallel
		for (int j = 1; j < child_pop_size; j++) {
			child_gen[j].fitness = get_fitness(child_gen[j].hat_size);
		}

		printf("CHILD GEN_____________\n");
		if (my_rank == 0) {
			cuda_print_parents(i, child_gen, child_pop_size, chrom_size);
		}
	}

	// MPI Barrier

	MPI_Barrier(MPI_COMM_WORLD);

	// Print whatever we are printing

	if (my_rank == 0) {
		cuda_print_parents(num_gens, parent_gen, pop_size, chrom_size);
	}

	// MPI Finalize

	MPI_Finalize();

	// Make call to cuda function to free memory

	//MORE FREEING NEEDED
	// cuda_free_gens();
	
	return 0;
}
