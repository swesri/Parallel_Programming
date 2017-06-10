CC = gcc
EXEC = jacobi_1D jacobi_2D  mat_vec 
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 

all: $(EXEC)
jacobi_1D: jacobi_1D.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ jacobi_1D.c  $(OBJS)

jacobi_2D: jacobi_2D.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ jacobi_2D.c $(OBJS)

mat_vec: mat_vec.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ mat_vec.c $(OBJS)

timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o $(OBJS) $(EXEC)
