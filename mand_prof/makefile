CC = gcc
EXEC = mandSEQ mandOMP mandSEQ_prof mandOMP_prof 
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3

all: $(EXEC)

mandSEQ: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ mandelbrot.c $(OBJS)

mandOMP: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ mandelbrot.c $(OBJS)

mandSEQ_prof: mand_prof.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ mand_prof.c $(OBJS) -DSEQUENTIAL

mandOMP_prof: mand_prof.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ mand_prof.c $(OBJS) -DOPENMP

timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
