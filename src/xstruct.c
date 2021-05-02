#include <stdio.h>

int test_int = 0;

typedef struct Degnome Degnome;
struct Degnome {
	double dna_array[test_int];
	double hat_size;

};

int main(int argc, char const *argv[]) {
	test_int = 0;
	printf("size of int %lu\n", sizeof(Degnome));
	test_int = 15;
	printf("size of int %lu\n", sizeof(Degnome));
}
