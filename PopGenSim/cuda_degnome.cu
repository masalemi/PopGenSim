#include <cuda.h>
#include <stdlib.h>

extern "C" {
    double* allocateDegnomes(unsigned int pop_size, unsigned int chrom_size);
    double* getDegnome(double* array, unsigned int chrom_size, unsigned int i);
    void freeDegnomes(double* array);
    double* allocateCudaArray(unsigned int size);
    void freeCudaArray(double* array);
}

double* allocateDegnomes(unsigned int pop_size, unsigned int chrom_size) {
    double* degnomes;
    cudaMallocManaged(&degnomes, pop_size * chrom_size * sizeof(double));
    return degnomes;
}

double* getDegnome(double* array, unsigned int chrom_size, unsigned int i) {
    return array + (chrom_size * i);
}

void freeDegnomes(double* array) {
    cudaFree(array);
}

double* allocateCudaArray(unsigned int size) {
    double* result;
    cudaMallocManaged(&result, size * sizeof(double));
    return result;
}

void freeCudaArray(double* array) {
    cudaFree(array);
}

