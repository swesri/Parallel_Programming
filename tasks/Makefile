CC = gcc
OBJS = timer.o
FLAGS = -fopenmp
DEBUGFLAG = -DDEBUG
SEQFLAGS = -O3 -lm

EXEC = Merge_sort Merge_sort_debug

all: $(EXEC)

Merge_sort: Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS)

Merge_sort_debug:Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS) $(DEBUGFLAG)  -o $@ $< $(OBJS)

timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)
