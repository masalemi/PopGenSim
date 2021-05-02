/**
 REMEMBER TO ADD COMMAND LINE ARGS ONCE THIS IS DONE!!!
 NOTE: it wouuld be very nice if this compartment had its own
 usage which could / would be called by anything that uses this

Also add command line args and testing for seeding and threading

*/

/**
@file fitfunc.c
@page fitfunc
@author Daniel R. Tabin
@brief Transforms fitness via multiple functions

*/

#include "device_fitfunc.h"
#include "string.h"
#include "math.h"
#include "stdlib.h"

__device__ fit_func_ptr func_to_run = &linear_returns;

void set_function(const char* func_name) {
	if (strcmp(func_name, "linear") == 0) {
		func_to_run = &linear_returns;
	}
	else if (strcmp(func_name, "sqrt") == 0){
		func_to_run = &sqrt_returns;
	}
	else if (strcmp(func_name, "close") == 0){
		func_to_run = &close_returns;
	}
	else if (strcmp(func_name, "ceiling") == 0){
		func_to_run = &ceiling_returns;
	}
	else if (strcmp(func_name, "log") == 0)
	{
		func_to_run = &logarithmic_returns;
	}
	else {
		func_to_run = &linear_returns;
	}
}

__device__ double linear_returns(double x) {
	return x;
}

__device__ double sqrt_returns(double x) {
	return sqrt(x);
}

__device__ double close_returns(double x) {
	return (target_num - fabs(target_num - x));
}

__device__ double ceiling_returns(double x) {
	if (x < target_num) {
		return x;
	}
	else {
		return (target_num) - 5 * fabs(target_num - x);
	}
}
__device__ double logarithmic_returns(double x) {
	return log(x);
}


__device__ double get_fitness(double hat_size) {
	return (*func_to_run)(hat_size);
}
