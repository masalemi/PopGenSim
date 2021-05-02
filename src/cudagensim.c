#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "flagparse.c"
#include "cuda_degnome.h"

// Declare all extern functions

extern void cuda_set_device(size_t myrank);
extern void* cuda_set_seed(unsigned long rng_seed, unsigned int pop_size);
extern void cuda_calloc_gens(unsigned int pop_size, unsigned int num_ranks, unsigned int chrom_size);
extern void kernel_launch();
extern void cuda_update_parents();
extern void cuda_print_parents(unsigned int num_gens);
extern void cuda_free_gens();
extern void cuda_free_rng(void* rng);

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

    chrom_size = flags[6];
    mutation_effect = flags[7];
    num_gens = flags[8];
    mutation_rate = flags[9];
    crossover_rate = flags[10];
    pop_size = flags[11];

    num_threads = flags[12];

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
    target_num = flags[14];

    if (flags[15] <= 0) {
        time_t currtime = time(NULL);                  // time
        unsigned long pid = (unsigned long) getpid();  // process id
        rngseed = currtime ^ pid;                      // random seed
    }
    else {
        rngseed = flags[15];
    }

    // make this command line args
    size_t threadsCount = 1;
    size_t blocksCount = 1;

    free(flags);

    // Set up MPI stuff (init and rank number)

    unsigned int my_rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Set CUDA Device Based on MPI rank

    cuda_set_device(my_rank);

    // calculate size of children generation
    int child_pop_size = pop_size / num_ranks;

    // Set random seed

    void* rng_ptr = cuda_set_seed(blocksCount, threadsCount, my_rank, rng_seed, child_pop_size);

    // Initialize local information and cuda calloc the memory we need (See lines 192 - 204)

    Degnome* parent_gen = Degnome_cuda_new(int pop_size, int chrom_size);
    Degnome* child_gen = Degnome_cuda_new(int child_pop_size, int chrom_size);

    Degnome_reorganize(Degnome* parent_gen, int pop_size, int chrom_size);
    Degnome_reorganize(Degnome* child_gen, int child_pop_size, int chrom_size);

    // Create buffer struct type (How to handle subarrays?)

    // const int nitems = 2;
    // int blocklengths[2] = {pop_size / num_ranks, 1};
    // MPI_datatype types[2] = {, MPI_DOUBLE};
    // MPI_datatype degnome_type;
    // MPI_Aint offsets[2];

    // offsets[0] = offsetof(Degnome, dna_array);
    // offsets[1] = offsetof(Degnome, hat_size);

    // MPI_Type_create_struct(nitems, blocklengths, offsets, types, &degnome_type);
    // MPI_Type_commit(&degnome_type);

    // Run generation simulation

    // get sizes of bytes to send
    int degnome_size = (sizeof(Degnome) + (chrom_size * sizeof(double));

    int send_bytes = child_pop_size*degnome_size;
    int recv_bytes = pop_size*degnome_size;

    for (int i = 0; i < num_gens; i++) {

        // Collect info from all other ranks to make a complete generation
        MPI_Allgather(child_gen, send_bytes, MPI_BYTE, parent_gen, recv_bytes, MPI_BYTE, MPI_COMM_WORLD);

        // get the pointers right
        Degnome_reorganize(blocksCount, threadsCount, parent_gen, pop_size, chrom_size);

        // make child generation 
        kernel_launch(parent_gen, child_gen, pop_size, child_pop_size, total_hat_size, cum_siz_arr, rng_ptr, blocksCount, threadsCount);
    }

    // MPI Barrier

    MPI_Barrier(MPI_COMM_WORLD);

    // Print whatever we are printing

    if (myrank == 0) {
        cuda_print_parents(num_gens);
    }

    // MPI Finalize

    MPI_Finalize();

    // Make call to cuda function to free memory

    cuda_free_gens()
    
    return 0;
}