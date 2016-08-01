// dst[i] = a[i] * b[i]
__kernel void
mul(__global float* __restrict  dst, __global float* __restrict  a, __global float* __restrict b, int N) {

	int i =  get_global_id(0);

	if(i < N) {
		dst[i] = a[i] * b[i];
	}
}

