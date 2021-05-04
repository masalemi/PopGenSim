#include <cuda.h>
#include <stdio.h>

extern "C" {
    void cudaSetup(int rank);
    void launchKernel(double* parent_gen_chrom, unsigned int parent_pop_size,
                      double* child_gen_chrom, unsigned int child_pop_size,
                      double* pair_array,
                      double* cumulative_array,
                      bool* crossovers, bool* mutations,
                      size_t blocks, size_t threads_per_block,
                      unsigned int chrom_size);
}

void cudaSetup(int rank) {
    // Set the cuda device based on rank index
    int cudaDeviceCount;
    cudaError_t cE;

    if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess) {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }

    int assignment = rank % cudaDeviceCount;
    if ((cE = cudaSetDevice(assignment)) != cudaSuccess) {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", rank, assignment, cE);
        exit(-1);
    }
}

__device__ void binarySearch(double value, int pop_size, double* search_arr, int* result) {
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

__device__ void transfer(double* child, double* parent, unsigned int len) {
    for (unsigned int i = 0; i < len; i++) {
        child[i] = parent[i];
    }
}


__device__ void degnomeMate(double* parent_gen_chrom,
                            int const parent1,
                            int const parent2,
                            double* child_gen_chrom,
                            int const i,
                            bool* crossovers, bool* mutations,
                            unsigned int chrom_size) {
    double* child = child_gen_chrom + (chrom_size * i);
    double* m_idx = parent_gen_chrom + (chrom_size * parent1);
    double* d_idx = parent_gen_chrom + (chrom_size * parent2);

    // TODO: Check if memcpy works
    transfer(child, m_idx, chrom_size);
    for (unsigned int loc = 0; loc < chrom_size; loc++) {
        unsigned int left = chrom_size - loc;
        if (crossovers[loc]) {
            if (loc & 1) {
                transfer(child + loc, d_idx + loc, left);
            } else {
                transfer(child + loc, m_idx + loc, left);
            }
        }
    }

    // TODO: Figure out randomness for this.
    double mutation = 5;
    for (unsigned int loc = 0; loc < chrom_size; loc++) {
        if (mutations[loc]) {
            child[loc] += mutation;
        }
    }
}


__global__ void kernelSelectMate(double* parent_gen_chrom, unsigned int parent_pop_size,
                                 double* child_gen_chrom, unsigned int child_pop_size,
                                 double* pair_array,
                                 double* cumulative_array,
                                 bool* crossovers, bool* mutations,
                                 unsigned int chrom_size) {
    unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int index = id; index < child_pop_size; index += stride) {
        unsigned int offset = 2 * index;
        double win_m = pair_array[offset];
        double win_d = pair_array[offset + 1];

        int m_index = 0;
        int d_index = 0;
        binarySearch(win_m, parent_pop_size, cumulative_array, &m_index);
        binarySearch(win_d, parent_pop_size, cumulative_array, &d_index);

        degnomeMate(parent_gen_chrom, m_index, d_index,
                    child_gen_chrom, index,
                    crossovers, mutations,
                    chrom_size);
    }
}

void launchKernel(double* parent_gen_chrom, unsigned int parent_pop_size,
                  double* child_gen_chrom, unsigned int child_pop_size,
                  double* pair_array,
                  double* cumulative_array,
                  bool* crossovers, bool* mutations,
                  size_t blocks, size_t threads_per_block,
                  unsigned int chrom_size) {
    kernelSelectMate<<<blocks, threads_per_block>>>(parent_gen_chrom, parent_pop_size,
                                                    child_gen_chrom, child_pop_size,
                                                    pair_array,
                                                    cumulative_array,
                                                    crossovers, mutations,
                                                    chrom_size);
    cudaDeviceSynchronize();
}

