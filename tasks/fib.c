#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int nfib(long n){
long i,j;
if(n==0) return 0;
else if(n==2 || n==1) return 1;
else{
#pragma omp task //shared(i) untied
{i=nfib(n-1);}
#pragma omp task     //s hared(j)
{j=nfib(n-2);}
#pragma omp taskwait
return i+j;
}}

int main(int argc, char **argv){
long n;
int v;
int count=100;
if(argc < 2){ printf("Usage: argv[0]\t N\n");exit(0);}

n=atoi(argv[1]);
if(n<1){ printf("n should be greater than 0\n"); exit(0);}
#pragma omp parallel
{
#pragma omp single
{
printf("I am inside\n");
#pragma omp task
{/*while(count--);*/printf("Task-1\n");}
#pragma omp task
{printf("Task-2\n");}
//#pragma omp taskwait
printf("pragma\n");
}
}
//#pragma omp parallel //shared (n,v)
//{
//#pragma omp single nowait
//v=nfib(n);
//}
//printf("%d\n",v);
return 0;
}
