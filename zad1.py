import pyopencl as cl
import pyopencl.array
import numpy as np
import time

# Dominik Brdar, Paralelno programiranje, vježba 3, 2022.
# zadatak 1)

if __name__ == '__main__':
    start_time = time.time()
    
    size = 2**15  # size of array of test numbersstart_time = time.time()
    G = size
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    input_arr = np.arange(1, size, dtype=np.int32)
    helper_arr = np.empty(size, dtype=np.int32)
	
    input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
    helper_buf = cl.Buffer(ctx, mf.READ_WRITE, helper_arr.nbytes)
    count = cl.Buffer(ctx, mf.READ_WRITE, 4)
    
    code = "".join(open("./zad1.cl", 'r').readlines())
    p = cl.Program(ctx, code).build()
    p.find_primes(queue, (G,), None, input_buf, helper_buf, count, np.int32(len(input_arr))).wait()
    cl.enqueue_copy(queue, helper_arr, helper_buf)
    
    print("Duration: %s seconds" % (time.time() - start_time))