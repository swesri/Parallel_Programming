#!/bin/bash
export MALLOC_CHECK_=4
#Include test cases for larger values of P and T
#Vertical only test
#T=1
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 12 -T 1 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 6  -T 1 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 11 -T 1 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 10 -T 1 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 7  -T 1 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 4  -T 1 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -x 11 -y 5  -T 1 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 5  -T 1 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -x 14 -y 5  -T 1 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 1  -T 1 -v #Num_tiles=p 
#T=2
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 12 -T 2 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 6  -T 2 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 11 -T 2 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 10 -T 2 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 7  -T 2 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 4  -T 2 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -x 11 -y 5  -T 2 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 5  -T 2 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -x 14 -y 5  -T 2 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 1  -T 2 -v #Num_tiles=p 
#T=3
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 12 -T 3 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 6  -T 3 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 11 -T 3 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 10 -T 3 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 7  -T 3 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 4  -T 3 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -x 11 -y 5  -T 3 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 5  -T 3 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -x 14 -y 5  -T 3 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -x 12 -y 1  -T 3 -v #Num_tiles=p 

#Horizontal only test
#T=1
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  12 -T 1 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  6  -T 1 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  11 -T 1 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  10 -T 1 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  7  -T 1 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  4  -T 1 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -y 11 -x  5  -T 1 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  5  -T 1 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -y 14 -x  5  -T 1 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  1  -T 1 -v #Num_tiles=p 
#T=2
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  12 -T 2 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  6  -T 2 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  11 -T 2 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  10 -T 2 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  7  -T 2 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  4  -T 2 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -y 11 -x  5  -T 2 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  5  -T 2 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -y 14 -x  5  -T 2 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  1  -T 2 -v #Num_tiles=p 
#T=3
mpirun -np 1  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  12 -T 3 -v #Num_Tiles=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  6  -T 3 -v #Num_tiles=2 Full tiles
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  11 -T 3 -v #Num_tiles=2 Partial tiles Size=1
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  10 -T 3 -v #Num_tiles=2 Partial tiles Size=2
mpirun -np 2  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  7  -T 3 -v #Num_tiles=2 Partial tiles Size>1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  4  -T 3 -v #Num_tiles=3 Full tiles
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 11 -y 11 -x  5  -T 3 -v #Num_tiles=3 Partial tiles Size=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  5  -T 3 -v #Num_tiles=3 Partial tiles Size=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 14 -y 14 -x  5  -T 3 -v #Num_tiles=3 Partial tiles Size>2
mpirun -np 12 Jacobi2D-BlockParallel-MPI -p 12 -y 12 -x  1  -T 3 -v #Num_tiles=p 

#Horizontal and Vertical test
#T=1
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  8  -T 1 -v #Prime number of processors 
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 8  -x  2  -T 1 -v #Odd but not prime number of processors
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  6  -T 1 -v #2*2 Grid of processors with Full tiles
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  7  -T 1 -v #2*2 Grid of processors with Partial tiles in X direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  6  -T 1 -v #2*2 Grid of processors with Partial tiles in Y direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  7  -T 1 -v #2*2 Grid of processors with Partial tiles in both direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  4  -T 1 -v #3*3 Grid of processors with Full tiles
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  5  -T 1 -v #3*3 Grid of processors with Partial tiles in X direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  4  -T 1 -v #3*3 Grid of processors with Partial tiles in Y direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  5  -T 1 -v #3*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  4  -T 1 -v #4*3 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  5  -T 1 -v #4*3 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  5  -T 1 -v #4*3 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  6  -T 1 -v #4*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  3  -T 1 -v #3*4 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 5  -x  4  -T 1 -v #3*4 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  3  -T 1 -v #3*4 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 6  -x  4  -T 1 -v #3*4 Grid of processors with Partial tiles in both direction

#T=2
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  8  -T 2 -v #Prime number of processors 
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 8  -x  2  -T 2 -v #Odd but not prime number of processors
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  6  -T 2 -v #2*2 Grid of processors with Full tiles
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  7  -T 2 -v #2*2 Grid of processors with Partial tiles in X direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  6  -T 2 -v #2*2 Grid of processors with Partial tiles in Y direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  7  -T 2 -v #2*2 Grid of processors with Partial tiles in both direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  4  -T 2 -v #3*3 Grid of processors with Full tiles
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  5  -T 2 -v #3*3 Grid of processors with Partial tiles in X direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  4  -T 2 -v #3*3 Grid of processors with Partial tiles in Y direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  5  -T 2 -v #3*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  4  -T 2 -v #4*3 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  5  -T 2 -v #4*3 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  5  -T 2 -v #4*3 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  6  -T 2 -v #4*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  3  -T 2 -v #3*4 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 5  -x  4  -T 2 -v #3*4 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  3  -T 2 -v #3*4 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 6  -x  4  -T 2 -v #3*4 Grid of processors with Partial tiles in both direction

#T=3
mpirun -np 3  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  8  -T 3 -v #Prime number of processors 
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 8  -x  2  -T 3 -v #Odd but not prime number of processors
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  6  -T 3 -v #2*2 Grid of processors with Full tiles
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 6  -x  7  -T 3 -v #2*2 Grid of processors with Partial tiles in X direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  6  -T 3 -v #2*2 Grid of processors with Partial tiles in Y direction
mpirun -np 4  Jacobi2D-BlockParallel-MPI -p 12 -y 7  -x  7  -T 3 -v #2*2 Grid of processors with Partial tiles in both direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  4  -T 3 -v #3*3 Grid of processors with Full tiles
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  5  -T 3 -v #3*3 Grid of processors with Partial tiles in X direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  4  -T 3 -v #3*3 Grid of processors with Partial tiles in Y direction
mpirun -np 9  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  5  -T 3 -v #3*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  4  -T 3 -v #4*3 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 3 -x  5  -T 3 -v #4*3 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  5  -T 3 -v #4*3 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 4  -x  6  -T 3 -v #4*3 Grid of processors with Partial tiles in both direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 4  -x  3  -T 3 -v #3*4 Grid of processors with Full tiles
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 5  -x  4  -T 3 -v #3*4 Grid of processors with Partial tiles in X direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 12 -y 5  -x  3  -T 3 -v #3*4 Grid of processors with Partial tiles in Y direction
mpirun -np 12  Jacobi2D-BlockParallel-MPI -p 15 -y 6  -x  4  -T 3 -v #3*4 Grid of processors with Partial tiles in both direction
