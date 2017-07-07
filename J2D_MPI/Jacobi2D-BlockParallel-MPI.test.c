/************************************************************************************************************************************************
 * Jacobi2D benchmark - MPI Parallelization                                                						 	*					
 *																	 	*
 * Usage:																 	*		
 * aprun -n8 Jacobi2D-BlockParallel-MPI 												 	*
 * For a run on 8 processes														 	*
 *																	 	*
 * Initial code: Provided in Fall 2015, CS475												 	*
 *																	 	*
 * Modified and removed bugs to work for partial tiles- Swetha Varadrajan Fall 2016 CS 475						 	*
 * Modifications made:															 	*
 * 	1. Made sure that there are no idle processors by checking the command line arguments. 						 	*
 *		ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_y))*(ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_x))) != p_count	*
 *	2. Removed dead-lock in the validation part of the program by modifying the upperbound of x and y tiles in case of partial tiles	*
 *		upperBound_x = (cmdLineArgs.problemSize - (tile_x*cmdLineArgs.tile_len_x))+1;							*
 *		upperBound_y = (cmdLineArgs.problemSize - tile_y*cmdLineArgs.tile_len_y)+1;							*
 *	3. Rectified the address calculation in the validation portion. This part fails in case of partial tiles because the tile length of	*
 *	   the partial tile was used instead of the tile length of the previous tile (tile index starts with 0) 				*
 *		int x_pos = (tile_x*cmdLineArgs.tile_len_x) + 1;										*
 *	4. Removed the 2 parameters (number of tiles in x and y direction) and replaced it with tile length in x and y direction.This will 	*
 *	   ensure correct validation in case of partial tiles. The corresponding code in util.h is also modified.				*
 *		verifyResultJacobi2DTiled(test_space,cmdLineArgs.problemSize,cmdLineArgs.globalSeed,cmdLineArgs.T,cmdLineArgs.tile_len_x,	*
 *         							cmdLineArgs.tile_len_y)								*
 ************************************************************************************************************************************************/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <getopt.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#define MIN(x,y) x<y?x:y
#define STENCIL(read,write,y,x) space[write][y][x] = \
                                                     ( space[read][y-1][x] +\
                                                       space[read][y][x] +\
                                                       space[read][y+1][x] +\
                                                       space[read][y][x+1] +\
                                                       space[read][y][x-1] )/5;


#include "util.h"

void packColToVec(double**data,int col,double*vec, int row_count){
  
	for(int i=1;i<=row_count;i++)
		vec[i-1]=data[i][col];
  return;
}
void unpackVecToCol(double**data,int col,double*vec, int row_count){
  
	for(int i=1;i<=row_count;i++)
		data[i][col]=vec[i-1];
  return;
}


int main( int argc, char* argv[] ){
  int p_count,rank;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &p_count );
  MPI_Status s; 
  setbuf(stdout, NULL);
  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);
  if(((ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_y))*(ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_x))) != p_count)
  {
	if(rank == 0) 
	  	fprintf(stderr,"Either idle processor or small tile size.\n");
        	MPI_Finalize();
    		return 0;
  	
  }
  int x_tile_count = ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_x);
  int y_tile_count = ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_y);
  int tile_x = rank % x_tile_count;
  int tile_y = rank / x_tile_count;
  int lowerBound_x = 1;
  int lowerBound_y = 1;
  int upperBound_x = lowerBound_x + cmdLineArgs.tile_len_x;
  int upperBound_y = lowerBound_y + cmdLineArgs.tile_len_y;

  if((tile_x*cmdLineArgs.tile_len_x)+upperBound_x>(cmdLineArgs.problemSize+1))
	upperBound_x = (cmdLineArgs.problemSize - (tile_x*cmdLineArgs.tile_len_x))+1;
  
  if((tile_y*cmdLineArgs.tile_len_y)+upperBound_y > cmdLineArgs.problemSize+1)
	upperBound_y = (cmdLineArgs.problemSize - tile_y*cmdLineArgs.tile_len_y)+1;
  
  int tile_len_x = upperBound_x - lowerBound_x;
  int tile_len_y = upperBound_y - lowerBound_y;
  double** space[2];
  int i; 

  space[0] = (double**)malloc((tile_len_y + 2) * sizeof(double*));
  space[1] = (double**)malloc((tile_len_y + 2) * sizeof(double*));
  if( space[0] == NULL || space[1] == NULL ){
    printf( "Could not allocate y axis of space array\n" );
    exit(0);
  }

  for( i = 0; i < tile_len_y+2; ++i ){
    space[0][i]=(double*)malloc((tile_len_x+2) * sizeof(double));
    space[1][i]=(double*)malloc((tile_len_x+2) * sizeof(double));
    if( space[0][i] == NULL || space[1][i] == NULL ){
      printf( "Could not allocate x axis of space array\n" );
      exit(0);
    }
  }
          
 
  srand(cmdLineArgs.globalSeed+rank);
  int x, y;
  
  for( y = lowerBound_y; y < upperBound_y; ++y )
    for( x = lowerBound_x; x < upperBound_x; ++x ){
      space[0][y][x] =rand() / (double)rand();
    }
  
  if(tile_x == 0 ){
    for( i = 0; i <= upperBound_y; ++i){
      space[0][i][0] = 0;
      space[1][i][0] = 0;
    }
  }
  if(tile_y == 0 ){
    for( i = 0; i <= upperBound_x; ++i){
      space[0][0][i] = 0;
      space[1][0][i] = 0;
    }
  }
  if(tile_x == (x_tile_count-1)){
    for( i = 0; i <= upperBound_y; ++i){
      space[0][i][upperBound_x] = 0;
      space[1][i][upperBound_x] = 0;
    }
  }
  if(tile_y == (y_tile_count-1)){
    for( i = 0; i <= upperBound_x; ++i){
      space[0][upperBound_y][i] = 0;
      space[1][upperBound_y][i] = 0;
    }
  }

  
  double* buffer1 = (double*)malloc((tile_len_y)*sizeof(double)); 
  double* buffer2 = (double*)malloc((tile_len_y)*sizeof(double)); 
  double* buffer3 = (double*)malloc((tile_len_y)*sizeof(double)); 
  double* buffer4 = (double*)malloc((tile_len_y)*sizeof(double)); 

  double start_time = MPI_Wtime();
  int t,read=0,write=1;
  int x_lb = 1;
  int y_lb = 1; 
  int x_ub = upperBound_x;
  int y_ub = upperBound_y;

  for( t = 0; t < cmdLineArgs.T; ++t )
  { 
	if(p_count>1)
	{
		if(y_tile_count > 1)
		{
			if(tile_y ==0) //send only south
			{
				MPI_Send(&space[read][upperBound_y-1][0],tile_len_x+2,MPI_DOUBLE,rank+x_tile_count,0, MPI_COMM_WORLD);
				MPI_Recv(&space[read][upperBound_y][0],tile_len_x+2, MPI_DOUBLE,rank+x_tile_count,0, MPI_COMM_WORLD,&s);
			}
			else if (tile_y == y_tile_count-1) //send only north
			{
				MPI_Recv(&space[read][0][0],tile_len_x+2,MPI_DOUBLE,rank-x_tile_count,0, MPI_COMM_WORLD,&s);				
				MPI_Send(&space[read][1][0],tile_len_x+2,MPI_DOUBLE,rank-x_tile_count,0, MPI_COMM_WORLD);
			}
			else if (0 < tile_y < y_tile_count-1) //send both south and north
			{
				MPI_Recv(&space[read][0][0],tile_len_x+2,MPI_DOUBLE,rank-x_tile_count,0, MPI_COMM_WORLD,&s);
				MPI_Send(&space[read][1][0],tile_len_x+2,MPI_DOUBLE,rank-x_tile_count,0, MPI_COMM_WORLD);
				MPI_Send(&space[read][upperBound_y-1][0],tile_len_x+2,MPI_DOUBLE,rank+x_tile_count,0, MPI_COMM_WORLD);
				MPI_Recv(&space[read][upperBound_y][0],tile_len_x+2, MPI_DOUBLE,rank+x_tile_count,0, MPI_COMM_WORLD,&s);
			}
		}
		if(x_tile_count > 1)
		{
			if(tile_x ==0) //send only east
			{
				packColToVec(space[read],upperBound_x-1,buffer1,tile_len_y);
				MPI_Send(buffer1,tile_len_y+2,MPI_DOUBLE,rank+1,1, MPI_COMM_WORLD);
				MPI_Recv(buffer2,tile_len_y+2,MPI_DOUBLE,rank+1,2, MPI_COMM_WORLD, &s);
				unpackVecToCol(space[read],upperBound_x,buffer2,tile_len_y);
			}
			else if (tile_x == x_tile_count-1) //send only west
			{
				MPI_Recv(buffer1,tile_len_y+2,MPI_DOUBLE,rank-1,1, MPI_COMM_WORLD, &s);
				unpackVecToCol(space[read],0,buffer1,tile_len_y);
				packColToVec(space[read],1,buffer2,tile_len_y);
				MPI_Send(buffer2,tile_len_y+2,MPI_DOUBLE,rank-1,2, MPI_COMM_WORLD);
			}
			else if (0 < tile_x < x_tile_count-1) //send both east and west
			{
				MPI_Recv(buffer3,tile_len_y+2,MPI_DOUBLE,rank-1,1, MPI_COMM_WORLD, &s);
				unpackVecToCol(space[read],0,buffer3,tile_len_y);
				packColToVec(space[read],1,buffer4,tile_len_y);
				MPI_Send(buffer4,tile_len_y+2,MPI_DOUBLE,rank-1,2, MPI_COMM_WORLD);
				
				packColToVec(space[read],upperBound_x-1,buffer1,tile_len_y);
				MPI_Send(buffer1,tile_len_y+2,MPI_DOUBLE,rank+1,1, MPI_COMM_WORLD);
				MPI_Recv(buffer2,tile_len_y+2,MPI_DOUBLE,rank+1,2, MPI_COMM_WORLD, &s);
				unpackVecToCol(space[read],upperBound_x,buffer2,tile_len_y);
				
			}
		}
	}



    
	for( y = lowerBound_y; y < upperBound_y; ++y )
	      for( x = lowerBound_x; x < upperBound_x; ++x )
		STENCIL( read, write, y, x);


	read = write;
	write = 1 - write;
  }
 
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  double time =  (end_time - start_time);


  // DO NOT EDIT CODE BELOW THIS LINE!!
  // 4 - output and optional verification
  if( cmdLineArgs.printtime && rank == 0 ){
    printf( "Time: %f\n", time );
  }

  if( cmdLineArgs.verify ){

    if(rank == 0){ 
      double** test_space;
      // allocate enough space for the entire problem size
      test_space = (double**)malloc((cmdLineArgs.problemSize+2) 
                       * sizeof(double*));
      if( test_space == NULL ){
        printf( "Could not allocate y axis of space array\n" );
        exit(0);
      }
  
      // allocate x index space
      for( i = 0; i < cmdLineArgs.problemSize+2; ++i ){
        test_space[i]=(double*)malloc((cmdLineArgs.problemSize+2) 
                         * sizeof(double));
        if( test_space[i] == NULL ){
          printf( "Could not allocate x axis of test_space array\n" );
          exit(0);
        }
      }
	
      int j;
      for( i=1; i<=tile_len_y;i++){
        for( j=1; j<=tile_len_x;j++){
          test_space[i][j] = space[read][i][j];
        }
      }
      
      int pid;
      for(pid = 1; pid<p_count; pid++){
        tile_x = pid % x_tile_count;
        tile_y = pid / x_tile_count;

        // the local (pe specific lower bound for the data indexes)
        lowerBound_x = tile_x*cmdLineArgs.tile_len_x;
        lowerBound_y = tile_y*cmdLineArgs.tile_len_y;
        upperBound_x = lowerBound_x + cmdLineArgs.tile_len_x;
        upperBound_y = lowerBound_y + cmdLineArgs.tile_len_y;

        // calculate a local tile length in case this is a partial tile
        if(upperBound_x > cmdLineArgs.problemSize){
          upperBound_x = cmdLineArgs.problemSize;
        }
        if(upperBound_y > cmdLineArgs.problemSize){
          upperBound_y = cmdLineArgs.problemSize;
        }
        tile_len_x = upperBound_x - lowerBound_x;
        tile_len_y = upperBound_y - lowerBound_y;
	int y_pos;
        int x_pos = (tile_x*cmdLineArgs.tile_len_x) + 1;
        for( i=1; i<=tile_len_y;i++){
          y_pos = lowerBound_y + i;
          MPI_Recv(&test_space[y_pos][x_pos],tile_len_x,
                   MPI_DOUBLE,pid,i,MPI_COMM_WORLD,&s);
        }
      } 
	
      if(!verifyResultJacobi2DTiled(test_space,
          cmdLineArgs.problemSize,
          cmdLineArgs.globalSeed,
          cmdLineArgs.T,
          cmdLineArgs.tile_len_x,
          cmdLineArgs.tile_len_y)){
        fprintf(stderr,"FAILURE\n");
      }else{
        fprintf(stderr,"SUCCESS\n");
      }
   // all of the other processors need to send
    }else{
      for( i=1; i<=tile_len_y;i++){
        MPI_Send(&space[cmdLineArgs.T&1][i][1],tile_len_x,MPI_DOUBLE,0,i,MPI_COMM_WORLD);
      }
    }
  }

  MPI_Finalize();

 
  
}
