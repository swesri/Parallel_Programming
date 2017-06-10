#set -o verbose
export OMP_NUM_THREADS=8

#Calculating execution time when #threads=8 
./jacobi_1D 4000 200000 | grep "time" | sed 's/Data size : 4000  , #iterations : 200000 , time : //' | sed 's/sec//' > time1.txt
#Calculating the speed-up
awk '{if (1.94/$1 >= 2)  
		print "Jacobi_1D Speed-up test on 8 threads pass";
	else
		print "Jacobi_1D Speed-up test on 8 threads fail. Should be atleast greater than 2 for Data size : 4000  , #iterations : 200000."; 
	}' time1.txt

#Calculating execution time when #threads=8 
./jacobi_2D 1000 2000 | grep "Time" | sed 's/Data : 1000 by 1000 , Iterations : 2000 , Time : //' | sed 's/sec//' > time2.txt
#Calculating the speed-up
awk '{if (4.72/$1 >= 3.3)  
		print "Jacobi_2D Speed-up test on 8 threads pass";
	else
		print "Jacobi_2D Speed-up test on 8 threads fail. Should be atleast greater than 3.3 for Data size : 1000  , #iterations : 2000."; 
	}' time2.txt

#Calculating execution time when #threads=8 
./mat_vec 25000 10000| grep "time" | sed 's/elapsed time = //'  > time3.txt
#Calculating the speed-up
awk '{if (0.28/$1 >= 3.1)  
		print "Mat_vec Speed-up test on 8 threads pass";
	else
		print "Mat_vec Speed-up test on 8 threads fail. Should be atleast greater than 3.1 for N=2500, M=10000 ."; 
	}' time3.txt

