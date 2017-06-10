///
/// matmultKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"
#include<stdio.h>
#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue = 0;
  float temp[2][2];
  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread copies just one element of shared_A and one element of shared_B
    
        shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
        shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

        // Synchronize to ensure all elements are read
        __syncthreads();
     
   
     // Do an inproduct of one row of shared_A and one col of shared_B
     // computing one Cvalue by accumulation
   // #pragma unroll
   int i=thread_row;
   int j=thread_col;
   //__shared__ float temp[2][2];
  int k=0;
  int m=0;
  int n,l; 
  // if((threadIdx.x==1)&&(threadIdx.y==1)&&(blockIdx.x==1)&&(blockIdx.y==1)) {
   for(int i=thread_row, n=0;n<=1;n++) {
     for(int j=thread_col,l=0;l<=1;l++) {
    for(int e=0; e<BLOCK_SIZE; ++e){
       Cvalue += shared_A[i][e] * shared_B[e][j];
     
      //printf("\t e=%d: %f",e,shared_A[i][e]);
    }
   //__syncthreads();
//   }
Csub[i*C.stride+j]=Cvalue;//temp[k][m]; 
    j=j+16;
   } //end of j
   i=i+16;
 } //end of i

 __syncthreads();
//}
}
 /*Csub[thread_row*C.stride+thread_col]=temp[0][0];//Cvalue;
 Csub[(thread_row+16)*C.stride+thread_col]=temp[1][0];
 Csub[thread_row*C.stride+(thread_col+16)]=temp[0][1];
 Csub[(thread_row+16)*C.stride+(thread_col+16)]=temp[1][1];*/
  



  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  
}

