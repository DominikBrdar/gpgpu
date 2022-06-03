#pragma comment(lib, "OpenCL.lib")
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "arraymalloc.h"
#include "boundary.h"
#include "jacobi.h"
#include "cfdio.h"
#include <CL/cl.hpp>

int main(int argc, char **argv)
{
	

	int printfreq=1000; //output frequency
	double error, bnorm;
	double tolerance=0.0; //tolerance for convergence. <=0 means do not check

	//main arrays
	double *psi;
	//temporary versions of main arrays
	double *psitmp;

	//command line arguments
	int scalefactor, numiter;

	//simulation sizes
	int bbase=10;
	int hbase=15;
	int wbase=5;
	int mbase=32;
	int nbase=32;

	int irrotational = 1, checkerr = 0;

	int m,n,b,h,w;
	int iter;
	int i,j;

	double tstart, tstop, ttot, titer;

	//do we stop because of tolerance?
	if (tolerance > 0) {checkerr=1;}

	//check command line parameters and parse them

	if (argc <3|| argc >4) {
		printf("Usage: cfd <scale> <numiter>\n");
		return 0;
	}

	scalefactor=atoi(argv[1]);
	numiter=atoi(argv[2]);

	if(!checkerr) {
		printf("Scale Factor = %i, iterations = %i\n",scalefactor, numiter);
	}
	else {
		printf("Scale Factor = %i, iterations = %i, tolerance= %g\n",scalefactor,numiter,tolerance);
	}

	printf("Irrotational flow\n");

	//Calculate b, h & w and m & n
	b = bbase*scalefactor;
	h = hbase*scalefactor;
	w = wbase*scalefactor;
	m = mbase*scalefactor;
	n = nbase*scalefactor;

	printf("Running CFD on %d x %d grid in serial\n",m,n);

	//allocate arrays
	psi    = (double *) malloc((m+2)*(n+2)*sizeof(double));
	psitmp = (double *) malloc((m+2)*(n+2)*sizeof(double));

	//zero the psi array
	for (i=0;i<m+2;i++) {
		for(j=0;j<n+2;j++) {
			psi[i*(m+2)+j]=0.0;
		}
	}

	//set the psi boundary conditions
	boundarypsi(psi,m,n,b,h,w);

	// get all platforms (drivers), e.g. NVIDIA
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	// use device[1] because that's a GPU; device[0] is the CPU
	cl::Device default_device = all_devices[1];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	// a context is like a "runtime link" to the device and platform;
	// i.e. communication is possible
	cl::Context context({ default_device });

	// create the program that we want to execute on the device
	cl::Program::Sources sources;

	// calculates for each element; C = A + B
	std::string kernel_code =
		"   void kernel jacobistep_p(global double* psinew, global double* psi, global const int* m, "
		"                          global const int* n) {"
		"       int ID, Nthreads, n, ratio, start, stop;"
		""
		"       ID = get_global_id(0);"
		"       Nthreads = get_global_size(0);"
		"       n = N[0];"
		""
		"       ratio = (n / Nthreads);"  // number of elements for each thread
		"       start = ratio * ID;"
		"       stop  = ratio * (ID + 1);"
		""
		"       for (int i=start; i<stop; i++) {"
		"			psinew[((i % m)+1)*(m+2)+ i / m +1]=0.25*(psi[(i%m)*(m+2)+i/m+1]+psi[(i%m+2)*(m+2)+i/m+1]+psi[(i%m+1)*(m+2)+i/m]+psi[(i%m+1)*(m+2)+i/m+2]);"
		"       }   "
		"   }";
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}

	cl::Buffer psi_new_buf(context, CL_MEM_READ_WRITE, (m + 2) * (n + 2) * sizeof(double));
	cl::Buffer psi_buf(context, CL_MEM_READ_WRITE, (m + 2) * (n + 2) * sizeof(double));

	// create a queue (a queue of commands that the GPU will execute)
	cl::CommandQueue queue(context, default_device);

	queue.enqueueWriteBuffer(psi_new_buf, CL_TRUE, 0, (m+2)*(n+2)*sizeof(double), psitmp);
    queue.enqueueWriteBuffer(psi_buf, CL_TRUE, 0, (m+2)*(n+2)*sizeof(double), psi);

	//compute normalisation factor for error
	bnorm=0.0;

	for (i=0;i<m+2;i++) {
			for (j=0;j<n+2;j++) {
			bnorm += psi[i*(m+2)+j]*psi[i*(m+2)+j];
		}
	}
	bnorm=sqrt(bnorm);

	//cl::KernelFunctor jacobistep_p(cl::Kernel(program, "jacobistep_p"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);

	cl::Kernel jacobistep_p(program, "jacobistep_p");
	

	//begin iterative Jacobi loop
	printf("\nStarting main loop...\n\n");
	tstart=gettime();

	for(iter=1;iter<=numiter;iter++) {

		//calculate psi for next iteration
		//jacobistep(psitmp,psi,m,n);
		//jacobistep_p(psitmp,psi,&m,&n);
		jacobistep_p.setArg(0, psitmp);
		jacobistep_p.setArg(1, psi);
		jacobistep_p.setArg(2, &m);
		jacobistep_p.setArg(3, &n);

		queue.enqueueNDRangeKernel(jacobistep_p, cl::NullRange, cl::NDRange(10), cl::NullRange);
		
	
		//calculate current error if required
		if (checkerr || iter == numiter) {
			error = deltasq(psitmp,psi,m,n);

			error=sqrt(error);
			error=error/bnorm;
		}

		//quit early if we have reached required tolerance
		if (checkerr) {
			if (error < tolerance) {
				printf("Converged on iteration %d\n",iter);
				break;
			}
		}

		//copy back 
		/**
		for(i=1;i<=m;i++) {
			for(j=1;j<=n;j++) {
				psi[i*(m+2)+j]=psitmp[i*(m+2)+j];
			}
		}
		**/

		queue.enqueueReadBuffer(psi_new_buf, CL_TRUE, 0, (m+2)*(n+2)*sizeof(double), psi);

		//print loop information
		if(iter%printfreq == 0) {
			if (!checkerr) {
				printf("Completed iteration %d\n",iter);
			}
			else {
				printf("Completed iteration %d, error = %g\n",iter,error);
			}
		}
	}	// iter

	queue.finish();

	if (iter > numiter) iter=numiter;

	tstop=gettime();

	ttot=tstop-tstart;
	titer=ttot/(double)iter;

	//print out some stats
	printf("\n... finished\n");
	printf("After %d iterations, the error is %g\n",iter,error);
	printf("Time for %d iterations was %g seconds\n",iter,ttot);
	printf("Each iteration took %g seconds\n",titer);

	//output results
	//writedatafiles(psi,m,n, scalefactor);
	//writeplotfile(m,n,scalefactor);

	//free un-needed arrays
	free(psi);
	free(psitmp);
	printf("... finished\n");

	return 0;
}
