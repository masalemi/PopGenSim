#include "jobqueue.h"
#include "degnome.h"
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>

typedef struct JobData JobData;
struct JobData {
	Degnome* child;
	Degnome* p1;
	Degnome* p2;
	gsl_rng* rng;
};

void usage(void);
int jobfunc(void *p);

double get_fitness(double hat_size);

const char *usageMsg =
    "Usage: polygensim [-c CL] [-p PS] [-g GE]  [-t] TH\n"
    "\n"
    "\"CL\" is chromosome length.  \"PS\" is the population size, \"GE\"\n"
    "is the number of generations that the simulation will run,\n"
    "and \"TH\" is the number of threads.  They can be in any\n"
    "order, and not all are needed.  Default CL is 50, default PS\n"
    "is 100, and the default G is 1000.\n";

pthread_mutex_t seedLock = PTHREAD_MUTEX_INITIALIZER;
unsigned long rngseed=0;

int pop_size;
int num_gens;

int num_threads;
int multi_threaded;

void usage(void) {
	fputs(usageMsg, stderr);
	exit(EXIT_FAILURE);
}

int jobfunc(void* p) {
    JobData* data = (JobData*) p;								//get data out
    Degnome_mate(data->child, data->p1, data->p2, data->rng);	//mate

    return 0;		//exited without error
}

double get_fitness(double hat_size){
	return hat_size;
}

int main(int argc, char **argv){

	chrom_size = 50;
	pop_size = 100;
	num_gens = 1000;

	num_threads = 1;
	multi_threaded = 0;


	if(argc > 9 || (argc%2) == 0){
		printf("\n");
		usage();
	}
	for(int i = 1; i < argc; i += 2){
		if(argv[i][0] == '-'){
			if (strcmp(argv[i], "-c") == 0){
				sscanf(argv[i+1], "%u", &chrom_size);
			}
			else if (strcmp(argv[i], "-p") == 0){
				sscanf(argv[i+1], "%u", &pop_size);
			}
			else if (strcmp(argv[i], "-g") == 0){
				sscanf(argv[i+1], "%u", &num_gens);
			}
			else if (strcmp(argv[i], "-t") == 0){
				sscanf(argv[i+1], "%u", &num_threads);
				multi_threaded = 1;
			}
			else{
				usage();
			}
		}
		else{
			usage();
		}
	}

    time_t currtime = time(NULL);                  // time
    unsigned long pid = (unsigned long) getpid();  // process id
    rngseed = currtime ^ pid;                      // random seed
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);    // rand generator

	Degnome* parents;
	Degnome* children;
	Degnome* temp;

	printf("%u, %u, %u\n", chrom_size, pop_size, num_gens);

	parents = malloc(pop_size*sizeof(Degnome));
	children = malloc(pop_size*sizeof(Degnome));

	for (int i = 0; i < pop_size; i++){
		parents[i].dna_array = malloc(chrom_size*sizeof(double));
		children[i].dna_array = malloc(chrom_size*sizeof(double));
		parents[i].hat_size = 0;

		for(int j = 0; j < chrom_size; j++){
			parents[i].dna_array[j] = (i+j);	//children isn't initiilized
			parents[i].hat_size += (i+j);
		}
	}

	printf("Generation 0:\n");
	for(int i = 0; i < pop_size; i++){
		printf("Degnome %u\n", i);
		for(int j = 0; j < chrom_size; j++){
			printf("%lf\t", parents[i].dna_array[j]);
		}
		printf("\nTOTAL HAT SIZE: %lg\n\n", parents[i].hat_size);
	}

	for(int i = 0; i < num_gens; i++){

		double fit = get_fitness(parents[0].hat_size);

		double total_hat_size = fit;
		double cum_hat_size[pop_size];
		cum_hat_size[0] = fit;

		for(int j = 1; j < pop_size; j++){
			fit = get_fitness(parents[j].hat_size);

			total_hat_size += fit;
			cum_hat_size[j] = (cum_hat_size[j-1] + fit);
		}

		for(int j = 0; j < pop_size; j++){

		    pthread_mutex_lock(&seedLock);
			gsl_rng_set(rng, rngseed);
		    rngseed = (rngseed == ULONG_MAX ? 0 : rngseed + 1);
    		pthread_mutex_unlock(&seedLock);

			int m, d;

			double win_m = gsl_rng_uniform(rng);
			win_m *= total_hat_size;
			double win_d = gsl_rng_uniform(rng);
			win_d *= total_hat_size;

			// printf("win_m:%lf, wind:%lf, max: %lf\n", win_m,win_d,total_hat_size);

			for (m = 0; cum_hat_size[m] < win_m; m++){
				continue;
			}

			for (d = 0; cum_hat_size[d] < win_d; d++){
				continue;
			}

			// printf("m:%u, d:%u\n", m,d);

			Degnome_mate(children + j, parents + m, parents + d, rng);		//Is selective
		}
		temp = children;
		children = parents;
		parents = temp;
	}

	printf("Generation %u:\n", num_gens);
	for(int i = 0; i < pop_size; i++){
		printf("Degnome %u\n", i);
		for(int j = 0; j < chrom_size; j++){
			printf("%lf\t", parents[i].dna_array[j]);
		}
		printf("\nTOTAL HAT SIZE: %lg\n\n", parents[i].hat_size);
	}

	//free everything

	for (int i = 0; i < pop_size; i++){
		free(parents[i].dna_array);
		free(children[i].dna_array);
		parents[i].hat_size = 0;
	}

	free(parents);
	free(children);

	gsl_rng_free (rng);
}