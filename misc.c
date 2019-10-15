/**
 * @file xdegnome.c
 * @author Daniel R. Tabin
 * @brief Misc functions
 */

#include "misc.h"
#include <stdlib.h>
#include <stdio.h>

int int_qsort_comparator(const void * ptra, const void * ptrb){
	int a = *((const int *)ptra);
	int b = *((const int *)ptrb);
	return (a - b);
}

void int_qsort(int * array, size_t num){
	qsort(array, num, sizeof(int), int_qsort_comparator);
}

//TODO -- refactor all instances of int_merge_sort to int_qsort.
void int_merge_sort(int* array, int min, int max){
	int_qsort(array, max+1);
}