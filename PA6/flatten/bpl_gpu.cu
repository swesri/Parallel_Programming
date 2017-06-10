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

double* d_A; 
double* d_B; 
double* d_C;
double* d_H;
double* d_Zh;
double* d_Zy;
double* d_P;
double* d_X;
double* d_Wxh;


// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

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
  double *dWxh; //Error Weights in between input and hidden layer = (N+1)*M
  double *dWhy; //Error Weights in between input and hidden layer = (M+1)*P

  double *Zh; //Weighted sum for hidden layer=I*M
  double *H;  // Activation values = I*(M+1)
  double *Zy; //Weighted sum for output layer=I*P 
  double *E;  //Calculated Errors = I*P
  double *P1; //Oredicted output = I*P
  double *P;  // (exp(Zy)) = I*P
  double *sum; //(summation of the P[i]s) = I
  
  double learningrate = 0.0001; /*learning rate */
  long b = cmdLineArgs.sample_per_iter;
  
  long k2 = cmdLineArgs.sample_total/b ; /*number of full bunches */
  long k3 = cmdLineArgs.sample_total-(k2*b); /* size of the partial bunch */

   dim3 dimGrid(b);                    
   dim3 dimBlock(cmdLineArgs.P); 
/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Memory allocations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
 
  inputs  = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.N);
  outputs = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.P);
  
  sum	  = (double*)malloc((b)*sizeof(double));

  /*for(long i = 0; i < cmdLineArgs.sample_total; ++i )
  {
	inputs[i] =(double*)malloc(cmdLineArgs.N * sizeof(double));
	outputs[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }*/

  Wxh     = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
  Why	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);
  dWxh    = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
  dWhy	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);

  /*for(long i = 0; i < cmdLineArgs.N+1; ++i )
  {
	Wxh[i] =(double*)malloc(cmdLineArgs.M * sizeof(double));	
	dWxh[i]=(double*)malloc(cmdLineArgs.M * sizeof(double));
  }

  for(long i = 0; i < cmdLineArgs.M+1; ++i )
  {
	Why[i] =(double*)malloc(cmdLineArgs.P * sizeof(double));
	dWhy[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }*/

  X	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.N+1));
  E	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  P	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  P1  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
  H	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M+1));
  Zh  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M));
  Zy  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));

  /*for(long i = 0; i < b; ++i )
  {
  X[i]	  = (double*)malloc((cmdLineArgs.N+1)*sizeof(double));
  E[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P1[i]   = (double*)malloc(cmdLineArgs.P*sizeof(double));
  H[i]	  = (double*)malloc((cmdLineArgs.M+1)*sizeof(double));
  Zh[i]	  = (double*)malloc(cmdLineArgs.M*sizeof(double));
  Zy[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  }*/

  if( inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
      || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P == NULL
	  || P1 == NULL || sum == NULL)
  {
    printf( "Could not allocate memory\n" );
    exit(0);
  }
size_t size = b * cmdLineArgs.P * sizeof(double);

   cudaError_t error;
       error = cudaMalloc((void**)&d_A, size);
	       if (error != cudaSuccess) Cleanup(false);
		       error = cudaMalloc((void**)&d_B, size);
			       if (error != cudaSuccess) Cleanup(false);
				       error = cudaMalloc((void**)&d_C, size);
					       if (error != cudaSuccess) Cleanup(false);
error = cudaMalloc((void**)&d_H, b * (cmdLineArgs.M+1) * sizeof(double));
           if (error != cudaSuccess) Cleanup(false);
		                  error = cudaMalloc((void**)&d_Zh, b * cmdLineArgs.M * sizeof(double));
						                     if (error != cudaSuccess) Cleanup(false);

error = cudaMalloc((void**)&d_P, b * (cmdLineArgs.P) * sizeof(double));
           if (error != cudaSuccess) Cleanup(false);
		                             error = cudaMalloc((void**)&d_Zy, b * cmdLineArgs.P * sizeof(double));
									                                              if (error != cudaSuccess) Cleanup(false);
printf("Hi\n");
error = cudaMalloc((void**)&d_X, b*(cmdLineArgs.N+1)*sizeof(double));
if(error!= cudaSuccess) Cleanup(false);
error = cudaMalloc((void**)&d_Wxh, cmdLineArgs.M*(cmdLineArgs.N+1)*sizeof(double));
if(error!= cudaSuccess) Cleanup(false);
//error = cudaMalloc((void**)&d_Zh, b*(cmdLineArgs.M)*sizeof(double));
//if(error!= cudaSuccess) Cleanup(false);

/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Initializations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  initializeW(Wxh,(cmdLineArgs.N+1),cmdLineArgs.M);
  initializeW(Why,(cmdLineArgs.M+1),cmdLineArgs.P);
  initializeI(inputs,cmdLineArgs.sample_total,cmdLineArgs.N);
  initializeO(outputs,cmdLineArgs.sample_total,cmdLineArgs.P);
//printf("Initialize over\n");
/*---------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------Training-------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
  initialize_timer();
  start_timer();
   //displayMatrix1 ("inputs", inputs, cmdLineArgs.sample_total, cmdLineArgs.N);
  for (long t=0; t<cmdLineArgs.iter; t++) //Time loop
  {
//  printf("Time loop:%ld\n",t);
	 for (long s=0; s<k2; s++) //Bunch loop
	  { 	
//	  printf("Bunch loop:%ld\n",s);
		for(long i=0;i<b;i++)
		{
		X(i,0)=H(i,0)=1;//bias setting
		//required input/output are copied from inputs/outputs to X and Y
	 	memcpy (&X(i,1), &inputs[cmdLineArgs.N*((s*b)+i)], cmdLineArgs.N*sizeof(double)); 
		}
		Y = &outputs[s*b*cmdLineArgs.P]; 
		 //displayMatrix1 ("expected input", X, b, cmdLineArgs.N+1);
		  //displayMatrix1 ("input/hidden weights", Wxy, , cmdLineArgs.P);
		/*Forward Phase*/
//		printf("Forward Phase\n");
//		mm(Zh,X,Wxh,b,cmdLineArgs.N+1,cmdLineArgs.M); //Zh=X*Wxh
//printf("Hello0\n");
error=cudaMemcpy(d_Wxh,Wxh,(cmdLineArgs.N+1)*cmdLineArgs.M*sizeof(double),cudaMemcpyHostToDevice);
if(error!=cudaSuccess)Cleanup(false);
error=cudaMemcpy(d_X,X,b*(cmdLineArgs.N+1)*sizeof(double),cudaMemcpyHostToDevice);
if(error!=cudaSuccess)Cleanup(false);
dim3 dimGri1(2,2);
   dim3 dimBlock1(2,2);
//printf("Hello1\n");
MatMulKernel<<<dimGri1,dimBlock1>>>(d_X,d_Wxh,d_Zh,b,cmdLineArgs.N+1,cmdLineArgs.M,0);
func(H,Zh,b,cmdLineArgs.M,1); //H=f1(Zh)
printf("Hello2\n");
error=cudaMemcpy(Zh,d_Zh,b*cmdLineArgs.M*sizeof(double),cudaMemcpyDeviceToHost);
if(error!=cudaSuccess)Cleanup(false);
displayMatrix1 ("weighted sum",Zh, b, cmdLineArgs.M);
//		 error = cudaMemcpy(d_Zh, Zh,b * cmdLineArgs.M * sizeof(double), cudaMemcpyHostToDevice);
		            // if (error != cudaSuccess) Cleanup(false);
/*	Activation<<<dimGrid, cmdLineArgs.M>>>(d_H,d_Zh);
		error = cudaGetLastError();
		             if (error != cudaSuccess) Cleanup(false);
					                  cudaThreadSynchronize();
		error = cudaMemcpy(H, d_H, b * (cmdLineArgs.M+1) * sizeof(double), cudaMemcpyDeviceToHost);
														                      if (error != cudaSuccess) Cleanup(false);
		//displayMatrix1 ("activation", H, b, cmdLineArgs.M+1);
*/		mm(Zy,H,Why,b,cmdLineArgs.M+1,cmdLineArgs.P); //Zy=H*Why	

func(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)	
/*		 error = cudaMemcpy(d_Zy, Zy,b * cmdLineArgs.P * sizeof(double), cudaMemcpyHostToDevice);
		                      if (error != cudaSuccess) Cleanup(false);
							          Exponents<<<dimGrid, cmdLineArgs.P>>>(d_P,d_Zy);
									          error = cudaGetLastError();
											                       if (error != cudaSuccess) Cleanup(false);
																cudaThreadSynchronize();
																										         error = cudaMemcpy(P, d_P, b * (cmdLineArgs.P) * sizeof(double), cudaMemcpyDeviceToHost);                                                                   if (error != cudaSuccess) Cleanup(false);
*/		reduction(P,sum,b,cmdLineArgs.P);  //summation of probabilities for each training sample
		prob(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)	
		//error(E,P1,Y,b,cmdLineArgs.P);	//E=P1-Y
		error = cudaMemcpy(d_C, Y, size, cudaMemcpyHostToDevice);
		    if (error != cudaSuccess) Cleanup(false);
			    error = cudaMemcpy(d_B, P1, size, cudaMemcpyHostToDevice);
				    if (error != cudaSuccess) Cleanup(false);
		 AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
		 error = cudaGetLastError();
		     if (error != cudaSuccess) Cleanup(false);
			     cudaThreadSynchronize();
				  error = cudaMemcpy(E, d_A, size, cudaMemcpyDeviceToHost);
				      if (error != cudaSuccess) Cleanup(false);

//displayMatrix1 ("expected error", E, b, cmdLineArgs.P);
		/*Backprpagation Phase*/
//		printf("Backward phase\n");
		mtm(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P); //dWhy=H'*E ('->transpose)		
		delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate); //Why=fn(dwhy)
		mmt(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P); //H=Why*E'		
		gradient_func(Zh,H,b,cmdLineArgs.M); //Zh=f1"(H) ("->gradient of f1)		
		mtm(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M);	//dWxh=X'Zh
		delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);//Wxh=fn(dWxh)
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
//		error(E,P1,Y,k3,cmdLineArgs.P);
			
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
//free(inputs);
//free(outputs);
//free(X);
//free(Zh);
//free(Zy);
//free(H);
//free(E);
//free(P);
//free(P1);
//free(sum);
//free(Wxh);
//free(Why);
//free(dWxh);
//free(dWhy);
/*-------------------------------------------------------END-----------------------------------------------------*/
return 0;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
	        
			    // Free device vectors
				    if (d_A)
					        cudaFree(d_A);
							    if (d_B)
								        cudaFree(d_B);
										    if (d_C)
											        cudaFree(d_C);

													    // Free host memory
																						    error = cudaThreadExit();
																										    
																											    if (!noError || error != cudaSuccess)
																												        printf("cuda malloc or cuda thread exit failed \n");
																														    
																															    fflush( stdout);
																																    fflush( stderr);

																																	    exit(0);
																																		}
void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
	    {
		      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
			        exit(-1);
					    }                         
						}
