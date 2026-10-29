Additional Materials for: " Default Risk and Income Fluctuations in Emerging Markets" by Cristina Arellano

This folder contains the codes and data used in the paper. 

CODE:

1. The main code is written in Fortran 90. It is compiled with Intel Fortran. It requires the IMSL library.
The Fortran code computes optimal policy functions. The following files are included and are required for the program: 

main.f90
parameters.f90
global.f90
process.f90

The main.f90 is the main program.


2. Simulations are written in Matlab. 
The Matlab code uploads the optimal policy functions and computes simulations, reports business cycles statistics and generates graphs.  
The following files are included and required for the program:

Policies_Simulation.m 
markovchain.m  (which is written by Mark Pederesen)


3. Additional files included:  
Stochastic process:  YMAT.dat and PIMAT.dat 
Detrended Argentina's GDP used in graph:  ydt.txt 


TO GENERATE RESULTS:
1. Save all the files in 1 folder. 

2. Run the Bond Program in Fortran. 
   	- Change the directory for uploading the stochastic process.  	
   	- Change the directory where the optimal policy vectors are saved.

2. Run the Policies_Simulation.m program in Matlab. 
	- This program uploads the saved vectors and runs simulations. 
	- It generates the statistics and figures of the paper. 
	- The program requires the file ydt.txt which is included in this folder.


DATA:

The excel file default_data.xls contains the raw data for the time series of GDP, consumption, spreads, and the trade balance (as % of GDP) for:
Argentina, Ecuador, and Russia.


 





 

