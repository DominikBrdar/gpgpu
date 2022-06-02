import pyopencl as cl, numpy as np

class PP:
    def __init__(self, path):
        self.memfag = cl.mem_flags
        self.context = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.context)
        self.code = "".join(open(path, 'r').readlines())
        self.program = cl.Program(self.context, self.code).build()

    def getQueue(self):
        return self.queue
    
    def getProgram(self):
        return self.program

    def getFlags(self):
        return self.memfag
    
    def getContext(self):
        return self.context


if __name__ == '__main__':

    # zadatak 1
    # get a program
    p = PP("./zad1.cl")
    # number of threads
    G = 7
    # number of "numbers" to check for primes
    n = 2000

    input_arr = np.array([i for i in range(n)], dtype=np.int32)
    helper_arr = np.empty(n, dtype=np.int32)

    buffer_input = cl.Buffer(p.getContext(), p.getFlags().READ_ONLY | p.getFlags().COPY_HOST_PTR, hostbuf=input_arr)
    buffer_helper = cl.Buffer(p.getContext(), p.getFlags().READ_WRITE, helper_arr.nbytes)

    p.getProgram().calc_primes(p.getQueue(), (G,), None, buffer_input, buffer_helper, np.int32(len(input_arr))).wait()
    cl.enqueue_copy(p.getQueue(), helper_arr, buffer_helper)


