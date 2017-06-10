__global__ void AddVectors(double** C, double** A,  double** B)
{
//printf("Hello\n");
	  int i= blockIdx.x;
      int j= threadIdx.x;
      C[i][j] = A[i][j];// - B[i][j];
	  printf("End of kernel\n");
printf("C[%d][%d]=%lf\n",i,j,C[i][j]);
return;
}
