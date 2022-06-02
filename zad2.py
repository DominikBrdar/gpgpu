import pyopencl as cl, numpy as np
import math, os, time

# Dominik Brdar, Paralelno programiranje, vježba 3, 2022.
# zadatak 2)

if __name__=='__main__':
	
	start_time = time.time()
	
	# get info of avalible platform and gpu
	platform = cl.get_platforms()
	gpu_devices = platform[1].get_devices()	
	print("Platforms: " + str(platform))
	print("OpenCL devices: " + str(gpu_devices))
	
	ctx = cl.Context([gpu_devices[0]])
	queue = cl.CommandQueue(ctx)
	
	cores = 0
	N = 10**7
	# Calculate pi in n cycles with opencl on specified number of cores. cores=0 => all cores
	p = cl.Program(ctx, """
		__kernel void calac_pi(__global float *result, const int n) {
	  		int gid = get_global_id(0);
	  		float x = (((float)gid - 0.5) / (float)n);
	  		result[gid] = 4.0 / (1.0 + x * x);
	}""").build()
	mf = cl.mem_flags
	r = np.zeros(n, dtype=np.float32)
	r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r.nbytes)
	
	p.calc_pi(queue, r.shape, None, r_buf, np.int32(N))
	cl.enqueue_copy(queue, r, r_buf)
	print("PI = " + str(r.sum()/N) + "\n n: " + str(N))
	
	print("Duration: %s seconds" % (time.time() - start_time))
	
