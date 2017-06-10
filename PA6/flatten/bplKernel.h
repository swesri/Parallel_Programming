__global__ void AddVectors(double* C, double* A,  double* B)
{
//printf("Hello\n");
	  int i= blockDim.x*blockIdx.x+threadIdx.x;

      C[i] = A[i] - B[i];
//	  printf("End of kernel\n");
//printf("C[%d][%d]=%lf\n",i,j,C[i][j]);
//return;
}

__global__ void Activation(double* C, double* A)
{
//printf("Hello\n");
      int i= blockDim.x*blockIdx.x+threadIdx.x;
//
if(threadIdx.x ==0)
	   C[i]=1;
else
		C[i] = tanh(A[i]) ;
//            //    printf("End of kernel\n");
//            //printf("C[%d][%d]=%lf\n",i,j,C[i][j]);
//            //return;
           }
__global__ void Exponents(double* C, double* A)
{
//printf("Hello\n");
      int i= blockDim.x*blockIdx.x+threadIdx.x;
//      //
//      if(threadIdx.x ==0)
  //           C[i]=1;
    //         else
                 C[i] = exp(A[i]) ;
//                     //            //    printf("End of kernel\n");
//                     //            //printf("C[%d][%d]=%lf\n",i,j,C[i][j]);
//                     //            //return;
                                }
