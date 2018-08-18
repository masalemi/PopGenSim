# Where the executable files will be copied
destination := $(HOME)/bin

#opt := -DNDEBUG -O3  -finline-functions  # For full optimization
opt :=  -O0 -fno-inline-functions      # For debugging

# Flags to determine the warning messages issued by the compiler

targets := polygensim

tests := xdegnome xjobqueue

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

all : $(targets)

test : $(tests)

# test polygensim.c
POLYGENSIM := polygensim.o degnome.o misc.o jobqueue.o
polygensim : $(POLYGENSIM)
	$(CC) $(CFLAGS) -o $@ $(POLYGENSIM) $(lib)

# test degnome.c
XDEGNOME := xdegnome.o degnome.o misc.o
xdegnome : $(XDEGNOME)
	$(CC) $(CFLAGS) -o $@ $(XDEGNOME) $(lib)

# test jobqueue.c
XJOBQUEUE := xjobqueue.o jobqueue.o
xjobqueue : $(XJOBQUEUE)
	$(CC) $(CFLAGS) -o $@ $(XJOBQUEUE) $(lib)

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

