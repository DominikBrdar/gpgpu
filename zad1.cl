// Dominik Brdar, Paralelno programiranje, vježba 3, 2022.
// zadatak 1)

int is_prim(int a) {
	if (a < 4) return 1;
	
    for (int i = 2; i < a; i++)
        if (a % i == 0) return 0;
        
    return 1;
}

__kernel void find_primes(__global int* input, __global int* output, __global int* count int arr_size) {
    __private uint gid = get_global_id(0);
    __private uint threads = get_local_size(0);

    for (int i = gid; i < arr_size; i += threads) 
        output[i] = is_prim(input[i]);
    
    atomic_add(count, 1);

    /*
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == 0) {
        volatile __local int count = 0;

        for(int i = 0; i < arr_size; i++) {
            printf("%d", input[i]);
            if (output[i]) {
            	printf(" is prime\n");
            	atomic_add(&count, 1);
            }
            else printf(" is not prime\n");
    */
        printf("Total number of primes = %d\n", count);
    }
}
