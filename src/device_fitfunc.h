#ifndef FITFUNC
#define FITFUNC

typedef double (*fit_func_ptr)(double);		//pointer to a fitness function which takes and returns a double

//char func_name[8];							//used for command line args (may not be needed on second thought delete later)
__device__ double input;
__device__ double target_num;

void set_function(const char*);
__device__ double get_fitness(double hat_size);


__device__ double linear_returns(double x);
__device__ double sqrt_returns(double x);
__device__ double close_returns(double x);
__device__ double ceiling_returns(double x);
__device__ double logarithmic_returns(double x);

#endif