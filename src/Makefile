# Where the executable files will be copied
destination := $(HOME)/bin

#opt := -DNDEBUG -O3  -finline-functions  # For full optimization
opt :=  -O0 -fno-inline-functions      # For debugging

# Flags to determine the warning messages issued by the compiler

targets := devosim polygensim genancesim

tests := xdegnome xfitfunc xjobqueue xmisc 

CC := gcc


warn := \
 -Wall \
 -Wcast-align \
 -Wcast-qual \
 -Wmissing-declarations \
 -Wmissing-prototypes \
 -Wnested-externs \
 -Wpointer-arith \
 -Wstrict-prototypes \
 -Wno-unused-parameter \
 -Wno-unused-function \
 -Wshadow \
 -Wundef \
 -Wwrite-strings

CFLAGS := -g -std=gnu99 $(warn) $(opt)
lib := -L/usr/local/lib -lgsl -lgslcblas -lpthread -lm

.c.o:
	$(CC) $(CFLAGS) $(incl) -c -o ${@F}  $<

pkg : $(targets)

all : $(targets) $(tests)

test : $(tests)

# run polygensim.c
DEVOSIM := devosim.o ance_degnome.o misc.o jobqueue.o fitfunc.o
devosim : $(DEVOSIM)
	$(CC) $(CFLAGS) -o $@ $(DEVOSIM) $(lib)
# run polygensim.c
POLYGENSIM := polygensim.o degnome.o misc.o jobqueue.o fitfunc.o
polygensim : $(POLYGENSIM)
	$(CC) $(CFLAGS) -o $@ $(POLYGENSIM) $(lib)

# run genancesim.c
GENANCESIM := genancesim.o degnome.o misc.o jobqueue.o fitfunc.o
genancesim : $(GENANCESIM)
	$(CC) $(CFLAGS) -o $@ $(GENANCESIM) $(lib)

# test fitfunc.c
XFITFUNC := xfitfunc.o fitfunc.o
xfitfunc : $(XFITFUNC)
	$(CC) $(CFLAGS) -o $@ $(XFITFUNC) $(lib)

# test degnome.c
XDEGNOME := xdegnome.o degnome.o misc.o
xdegnome : $(XDEGNOME)
	$(CC) $(CFLAGS) -o $@ $(XDEGNOME) $(lib)

# test jobqueue.c
XJOBQUEUE := xjobqueue.o jobqueue.o
xjobqueue : $(XJOBQUEUE)
	$(CC) $(CFLAGS) -o $@ $(XJOBQUEUE) $(lib)

#test misc.c
XMISC := xmisc.o misc.o
xmisc : $(XMISC)
	$(CC) $(CFLAGS) -o $@ $(XMISC) $(lib)

# Make dependencies file
depend : *.c *.h
	echo '#Automatically generated dependency info' > depend
	$(CC) -MM $(incl) *.c >> depend

clean :
	rm -f *.a *.o *~ 

include depend

.SUFFIXES:
.SUFFIXES: .c .o
.PHONY: clean

cuda: cudagensim.* cuda_degnome.* fitfunc.*
	mpixlc -g cudagensim.c -c -o cudagensim-mpi.o
	nvcc -g -G -arch=sm_70 cudagensim.cu -c -o cudagensim-cuda.o
	nvcc -g -G -arch=sm_70 cuda_degnome.cu -c -o cuda_degnome.o
	# nvcc -g -G -arch=sm_70 device_fitfunc.cu -c -o fitfunc.o
	gcc -g -std=gnu99 -Wall -Wcast-align -Wcast-qual -Wmissing-declarations -Wmissing-prototypes -Wnested-externs -Wpointer-arith -Wstrict-prototypes -Wno-unused-parameter -Wno-unused-function -Wshadow -Wundef -Wwrite-strings -O0 -fno-inline-functions -c -o flagparse.o flagparse.c
	gcc -g -std=gnu99 -Wall -Wcast-align -Wcast-qual -Wmissing-declarations -Wmissing-prototypes -Wnested-externs -Wpointer-arith -Wstrict-prototypes -Wno-unused-parameter -Wno-unused-function -Wshadow -Wundef -Wwrite-strings -O0 -fno-inline-functions -c -o fitfunc.o fitfunc.c
	mpicc -g cudagensim-mpi.o cudagensim-cuda.o cuda_degnome.o fitfunc.o flagparse.o -o cudagensim.exe \
		-L/usr/local/cuda-10.2/lib64/ -lcudadevrt -lcudart -lstdc++
