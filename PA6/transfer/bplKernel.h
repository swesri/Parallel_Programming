#define learning_rate 0.0001

__global__ void Activation(double* C, double* A)
{
	int i= blockDim.x*blockIdx.x+threadIdx.x;
	int j= (blockDim.x-1)*blockIdx.x+threadIdx.x-1;
	if(threadIdx.x ==0)
	   	C[i]=1;
	else
		C[i] = tanh(A[j]);
}


__global__ void Exponents(double* C, double* A)
{
      int i= blockDim.x*blockIdx.x+threadIdx.x;
      C[i] = exp(A[i]) ;
}

__global__ void Division(double* C, double* A, double* B)
{
      int i= blockDim.x*blockIdx.x+threadIdx.x;
      int j= blockIdx.x;
      C[i] = A[i]/B[j] ;
}

__global__ void ErrorCalc(double* C, double* A,  double* B)
{
	int i= blockDim.x*blockIdx.x+threadIdx.x;
	C[i] = A[i] - B[i];
}

__global__ void Subtraction(double* C, double* A)
{
      int i= blockDim.x*blockIdx.x+threadIdx.x;
      A[i] = A[i]-(C[i]*learning_rate) ;
}

__global__ void Gradient(double* C, double* A)
{
      int i= blockDim.x*blockIdx.x+threadIdx.x;
      int j= (blockDim.x+1)*blockIdx.x+threadIdx.x+1;
	  C[i] = A[j]*(1-pow(tanh(C[i]),(double)2));
}

__global__ void Reduction(double* C, double* A, long P)
{
      int i= blockIdx.x;
	A[i]=0;
      for(int j=0;j<P;j++)
	A[i] += C[j];
}
