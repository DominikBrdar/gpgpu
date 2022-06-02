import pyopencl as cl, numpy as np

# Dominik Brdar, Paralelno programiranje, vježba 3, 2022.
# zadatak 1)

size = 200000

input_arr = np.arange(1, size, dtype=np.int32)

ctx = cl.create_some_context
queue = cl.CommandQueue(ctx)

buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input_arr.nbytes)
cl.enqueue_write_buffer(queue, buf, input_arr);

threads = 7

prog = cl.Program(ctx, """
	__kernel void find_primes(__global int* input, int arr_size) {
    	__private uint gid = get_global_id(0);
    	__private uint threads = get_local_size(0);

    	for (int i = gid; i < arr_size; i += threads) {
        	output[i] = 1;
        	if (input[i] > 3) {
    			for (int i = 2; i < a; i++)
        			if (a % i == 0) 
        				output[i] = 0;
        	}
		}
    	barrier(CLK_LOCAL_MEM_FENCE);

		if (gid == 0) {
    		int count = 0;

        	for(int i = 0; i < arr_size; i++) {
            	printf("%d", input[i]);
            	if (output[i]) {
            		printf(" is prime\n");
            		count++;
            	}
            	else printf(" is not prime\n")
        	}
        	printf("Total number of primes = %d", count);
    	}
	} """).build()

prog.find_primes(queue, input_arr.shape, (threads,), buf, size)

