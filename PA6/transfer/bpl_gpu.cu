/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "timer.h"
#include "util.h"
#include "bunch-new.h"
#include "bplKernel.h"
#include "matmultKernel00.cu"

#define X(i,j) X[((i)*(cmdLineArgs.N+1))+(j)]
#define H(i,j) H[((i)*(cmdLineArgs.M+1))+(j)]

double* d_E; 
double* d_P1; 
double* d_P;
double* d_H;
double* d_Zh;
double* d_Zy;
double* d_X;
double* d_Y;
double* d_Wxh;
double* d_Why;
double* d_dWxh;
double* d_dWhy;
double* d_sum;


// Utility Functions
void Cleanup(bool);
void checkCUDAerrors(const char *msg);

int main(int argc, char** argv) 
{

/*---------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------Command line parsing--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Variable Declaration------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  /*Array description and its size in the comments next to its declation*/

  double *inputs;//Given inputs = total number of samples(S)*number of inputs per sample(N) 
  double *outputs;//Expected outputs = total number of samples(S)*number of outputs per sample(P) 

  double *X;//Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
  double *Y;//Output for a given iteration = bunch size(I)*number of outputs per sample(P)

  double *Wxh; //Weights in between input and hidden layer = (N+1)*M
  double *Why; //Weights in between input and hidden layer = (M+1)*P
  double *dWxh; //errors Weights in between input and hidden layer = (N+1)*M
  double *dWhy; //errors Weights in between input and hidden layer = (M+1)*P

  double *Zh; //Weighted sum for hidden layer=I*M
  double *H;  // Activation values = I*(M+1)
  double *Zy; //Weighted sum for output layer=I*P 
  double *E;  //Calculated errorss = I*P
  double *P1; //Oredicted output = I*P
  double *P;  // (exp(Zy)) = I*P
  double *sum; //(summation of the P[i]s) = I
  
  double learningrate = 0.0001; /*learning rate */
  long b = cmdLineArgs.sample_per_iter;
  
  long k2 = cmdLineArgs.sample_total/b ; /*number of full bunches */
  long k3 = cmdLineArgs.sample_total-(k2*b); /* size of the partial bunch */
 
/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Memory allocations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
 
  inputs  = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.N);
  outputs = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.P);
  
  sum	  = (double*)malloc((b)*sizeof(double));

  Wxh     = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
  Why	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);
  dWxh    = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
  dWhy	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);

  X	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.N+1));
  E	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  P	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  P1  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  H	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M+1));
  Zh  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M));
  Zy  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));

  if( inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
      || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P == NULL
	  || P1 == NULL || sum == NULL)
  {
    printf( "Could not allocate memory\n" );
    exit(0);
  }
   size_t size = b * cmdLineArgs.P * sizeof(double);

   cudaError_t errors;
   errors = cudaMalloc((void**)&d_E, size);
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_Y, size);
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_P1, size);
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_P, size);
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_Zy, size);
   if (errors != cudaSuccess) Cleanup(false);

   errors = cudaMalloc((void**)&d_H, b * (cmdLineArgs.M+1) * sizeof(double));
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_Zh, b * cmdLineArgs.M * sizeof(double));
   if (errors != cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_X, b*(cmdLineArgs.N+1)*sizeof(double));
   if(errors!= cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_sum, b * sizeof(double));
   if (errors != cudaSuccess) Cleanup(false);

   errors = cudaMalloc((void**)&d_Wxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double));
   if(errors!= cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_dWxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double));
   if(errors!= cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_Why, cmdLineArgs.P*(cmdLineArgs.M+1)*sizeof(double));
   if(errors!= cudaSuccess) Cleanup(false);
   errors = cudaMalloc((void**)&d_dWhy, cmdLineArgs.P*(cmdLineArgs.M+1)*sizeof(double));
   if(errors!= cudaSuccess) Cleanup(false);

/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Initializations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  initializeW(Wxh,(cmdLineArgs.N+1),cmdLineArgs.M);
  initializeW(Why,(cmdLineArgs.M+1),cmdLineArgs.P);
  initializeI(inputs,cmdLineArgs.sample_total,cmdLineArgs.N);
  initializeO(outputs,cmdLineArgs.sample_total,cmdLineArgs.P);

/*---------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------Training-------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
  

dim3 dimGrid(2,2);                    
dim3 dimBlock(2,2);

	errors = cudaMemcpy(d_Wxh, Wxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_Zh, Zh, b * (cmdLineArgs.M) * sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_H, H, b * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_Why, Why, cmdLineArgs.P*(cmdLineArgs.M+1)*sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_Zy, Zy, size, cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_P, P, size, cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_P1, P1, size, cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_E, E, size, cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_dWhy, dWhy, cmdLineArgs.P*(cmdLineArgs.M+1)*sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_dWxh, dWxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);
	errors = cudaMemcpy(d_sum, sum, b*sizeof(double), cudaMemcpyHostToDevice);
    	if (errors != cudaSuccess) Cleanup(false);

initialize_timer();
start_timer();		

  for (long t=0; t<cmdLineArgs.iter; t++) //Time loop
  {
 	for (long s=0; s<k2; s++) //Bunch loop
	  { 	
		for(long i=0;i<b;i++)
		{
			X(i,0)=H(i,0)=1;//bias setting
			memcpy (&X(i,1), &inputs[cmdLineArgs.N*((s*b)+i)], cmdLineArgs.N*sizeof(double)); 
		}
		Y = &outputs[s*b*cmdLineArgs.P]; 
		errors = cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
    		if (errors != cudaSuccess) Cleanup(false);
		errors = cudaMemcpy(d_X, X, b*(cmdLineArgs.N+1)*sizeof(double), cudaMemcpyHostToDevice);
    		if (errors != cudaSuccess) Cleanup(false);
		
		//mm(Zh,X,Wxh,b,cmdLineArgs.N+1,cmdLineArgs.M); //Zh=X*Wxh
		MatrixMult<<<dimGrid,dimBlock>>>(d_X,d_Wxh,d_Zh,b,cmdLineArgs.N+1,cmdLineArgs.M,0);
		errors = cudaMemcpy(Zh, d_Zh, b * (cmdLineArgs.M)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
		displayMatrix1 ("weighted sum", Zh, b, cmdLineArgs.M);

		//func(H,Zh,b,cmdLineArgs.M,1);
		//errors = cudaMemcpy(d_Zh, Zh, b * (cmdLineArgs.M) * sizeof(double), cudaMemcpyHostToDevice);
    	//	if (errors != cudaSuccess) Cleanup(false);
		errors = cudaMemcpy(d_H, H, b * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyHostToDevice);
		if (errors != cudaSuccess) Cleanup(false);
		Activation<<<b, cmdLineArgs.M+1>>>(d_H,d_Zh);
		errors = cudaMemcpy(H, d_H, b * (cmdLineArgs.M+1)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
		mm(Zy,H,Why,b,cmdLineArgs.M+1,cmdLineArgs.P); //Zy=H*Why
		//displayMatrix1("Activation", H, b, cmdLineArgs.M+1);
	
		//func(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)
		errors = cudaMemcpy(d_Zy, Zy, size, cudaMemcpyHostToDevice);
    		if (errors != cudaSuccess) Cleanup(false);
		Exponents<<<dimGrid, cmdLineArgs.P>>>(d_P,d_Zy);
		errors = cudaMemcpy(P, d_P, b * (cmdLineArgs.P)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
								                                                              
		//reduction(P,sum,b,cmdLineArgs.P);  
		Reduction<<<b, 1>>>(d_P,d_sum,cmdLineArgs.P);
		errors = cudaMemcpy(sum, d_sum, b*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);

		//prob(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)
		errors = cudaMemcpy(d_sum, sum, b*sizeof(double), cudaMemcpyHostToDevice);
		if (errors != cudaSuccess) Cleanup(false);
		Division<<<b, cmdLineArgs.P>>>(d_P1,d_P,d_sum);
		errors = cudaMemcpy(P1, d_P1, b * (cmdLineArgs.P)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
	
		//error(E,P1,Y,b,cmdLineArgs.P);	//E=P1-Y
		ErrorCalc<<<b, cmdLineArgs.P>>>(d_E,d_P1,d_Y);
		errors = cudaMemcpy(E, d_E, size, cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
		
		//MatrixMult<<<>>>();
		//Activation<<<b, cmdLineArgs.M+1>>>(d_H,d_Zh);
		//MatrixMult<<<>>>();
		//Exponents<<<dimGrid, cmdLineArgs.P>>>(d_P,d_Zy);
		//Reduction<<<b, 1>>>(d_P,d_sum,cmdLineArgs.P);
		//Division<<<b, cmdLineArgs.P>>>(d_P1,d_P,d_sum);
		//ErrorCalc<<<b, cmdLineArgs.P>>>(d_E,d_P1,d_Y);
		
		/*Backprpagation Phase*/
		mtm(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P); //dWhy=H'*E ('->transpose)		
		//delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate); //Why=fn(dwhy)
		errors = cudaMemcpy(d_Why, Why, cmdLineArgs.P * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyHostToDevice);
    		if (errors != cudaSuccess) Cleanup(false);
		errors = cudaMemcpy(d_dWhy, dWhy, cmdLineArgs.P * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyHostToDevice);
		if (errors != cudaSuccess) Cleanup(false);
		Subtraction<<<cmdLineArgs.P, cmdLineArgs.M+1>>>(d_dWhy,d_Why);
		errors = cudaMemcpy(Why, d_Why, cmdLineArgs.P * (cmdLineArgs.M+1)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);

		mmt(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P); //H=Why*E'		
		//gradient_func(Zh,H,b,cmdLineArgs.M); //Zh=f1"(H) ("->gradient of f1)	
		errors = cudaMemcpy(d_H, H, b * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyHostToDevice);
		if (errors != cudaSuccess) Cleanup(false);
		Gradient<<<b, cmdLineArgs.M>>>(d_Zh,d_H);
		errors = cudaMemcpy(Zh, d_Zh, b * (cmdLineArgs.M)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);
		//displayMatrix1(“Gradient”, Zh, b, cmdLineArgs.M);

	
		mtm(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M);	//dWxh=X'Zh
		//delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);//Wxh=fn(dWxh)
		errors = cudaMemcpy(d_Wxh, Wxh, cmdLineArgs.M * (cmdLineArgs.N+1) * sizeof(double), cudaMemcpyHostToDevice);
    		if (errors != cudaSuccess) Cleanup(false);
		errors = cudaMemcpy(d_dWxh, dWxh, cmdLineArgs.M * (cmdLineArgs.N+1) * sizeof(double), cudaMemcpyHostToDevice);
		if (errors != cudaSuccess) Cleanup(false);
		Subtraction<<<cmdLineArgs.M, cmdLineArgs.N+1>>>(d_dWxh,d_Wxh);
		errors = cudaMemcpy(Wxh, d_Wxh, cmdLineArgs.M * (cmdLineArgs.N+1)*sizeof(double), cudaMemcpyDeviceToHost);
		if (errors != cudaSuccess) Cleanup(false);

		//MatrixMult<<<>>>();
		//Subtraction<<<cmdLineArgs.P, cmdLineArgs.M+1>>>(d_dWhy,d_Why);
		//MatrixMult<<<>>>();
		//Gradient<<<b, cmdLineArgs.M>>>(d_Zh,d_H);
		//MatrixMult<<<>>>();
		//Subtraction<<<cmdLineArgs.M, cmdLineArgs.N+1>>>(d_dWxh,d_Wxh);

	}
	if(k3)
	{
		for(long i=0;i<k3;i++)
		{
		X(i,0)=H(i,0)=1;
	 	memcpy (&X(i,1), &inputs[cmdLineArgs.N*((k2*b)+i)], cmdLineArgs.N*sizeof(double));
		}
		Y = &outputs[k2*b*cmdLineArgs.P];

		/*Forward Phase*/
		mm(Zh,X,Wxh,k3,cmdLineArgs.N+1,cmdLineArgs.M);
		func(H,Zh,k3,cmdLineArgs.M,1);
		mm(Zy,H,Why,k3,cmdLineArgs.M+1,cmdLineArgs.P);		
		func(P,Zy,k3,cmdLineArgs.P,0); 
		reduction(P,sum,k3,cmdLineArgs.P);  
		prob(P,P1,sum,k3,cmdLineArgs.P);  
		error(E,P1,Y,k3,cmdLineArgs.P);
			
		/*Backprpagation Phase*/ 		
		mtm(dWhy,H,E,cmdLineArgs.M+1,k3,cmdLineArgs.P);
		delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate);
		mmt(H,Why,E,k3,cmdLineArgs.M+1,cmdLineArgs.P);		
		gradient_func(Zh,H,k3,cmdLineArgs.M);		
		mtm(dWxh,X,Zh,cmdLineArgs.N+1,k3,cmdLineArgs.M);
		delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);

	}	
   }

  stop_timer();
  double time = elapsed_time();
  printf( "Time: %lf\n",time);

/*error = cudaMemcpy(Wxh, d_Wxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double), cudaMemcpyDeviceToHost);
if (error != cudaSuccess) Cleanup(false);
error = cudaMemcpy(Why, d_Why, cmdLineArgs.P*(cmdLineArgs.M+1)*sizeof(double), cudaMemcpyDeviceToHost);
if (error != cudaSuccess) Cleanup(false);*/
	
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Print outputs----------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
   if(cmdLineArgs.V)
   {
	/*Need the following 2 statements for Testing*/
	displayMatrix1 ("input/hidden weights", Wxh, cmdLineArgs.N+1, cmdLineArgs.M);
	displayMatrix1 ("hidden/output weights", Why, cmdLineArgs.M+1, cmdLineArgs.P);
	/* Useful for analyzing the accuracy of prediction */
	/*if(k3)
	{	
		displayVector ("last input", &X[k3-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[k3-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[k3-1], cmdLineArgs.P);
	}
	else
	{
		displayVector ("last input", &X[b-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[b-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[b-1], cmdLineArgs.P);
	}*/
   }
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Free Memory------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
free(inputs);
free(outputs);
free(X);
free(Zh);
free(Zy);
free(H);
free(E);
free(P);
free(P1);
free(sum);
free(Wxh);
free(Why);
free(dWxh);
free(dWhy);
/*-------------------------------------------------------END-----------------------------------------------------*/
return 0;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_E)
        cudaFree(d_E);
    if (d_P)
        cudaFree(d_P);
    if (d_P1)
        cudaFree(d_P1);

  
        
    error = cudaThreadExit();
    
    if (!noError || error != cudaSuccess)
{        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}
