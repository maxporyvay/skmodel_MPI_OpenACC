ARCH=sm_60
HOST_COMP=mpicc
CC = pgcc
CCFLAGS = -O2 -lm -L/opt/open_mpi/lib -lmpi -I/opt/open_mpi/include
ACCFLAGS = -acc -ta=tesla:cc60,time -Minfo=accel

all: gpu_task.o

gpu_task.o: gpu_task.c
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $@ $<
