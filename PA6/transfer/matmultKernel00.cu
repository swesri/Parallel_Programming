
//type0:MM,type1:MTM,type2:MMT
#define by 2
#define bx 2
__global__ void MatrixMult(double* A, double* B, double* C, long N, long M, long P, int type){

 
  double *Asub, *Bsub, *Csub;

  int thread_col = threadIdx.y;
  int thread_row = threadIdx.x;
  int block_col = blockIdx.y;
  int block_row = blockIdx.x;

  Csub = &C[P * blockDim.y * block_row + blockDim.x * block_col];

  double Cvalue = 0;

if(type==0)
{
  for (int m = 0;  m < (M / blockDim.x); ++m){
    
    Asub = &A[M * blockDim.y * block_row + blockDim.x * m];
    Bsub = &B[M * blockDim.x * block_col + blockDim.y * m];


    __shared__ double shared_A[by][bx];
    __shared__ double shared_B[bx][by];

  
    shared_A[thread_row][thread_col] = Asub[thread_row * M/*blockDim.x*/ + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * M/*blockDim.y*/ + thread_col];

   
    __syncthreads();

    
#pragma unroll
    for(int e=0; e<blockDim.x; ++e)
       Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];

    __syncthreads();
  }
}

//printf("(%d,%d)=%d\n",blockIdx.x,blockIdx.y,thread_row*blockDim.x+thread_col);
//printf("ThreadIds=%d,%d\n",threadIdx.x,threadIdx.y);
//printf("global=%d\tlocal=%d\tvalue=%lf\n",(P * blockDim.y * block_row + blockDim.x * block_col),(thread_row * blockDim.x + thread_col),Cvalue);
Csub[thread_row * P/*blockDim.x*/ + thread_col] = Cvalue;
}

