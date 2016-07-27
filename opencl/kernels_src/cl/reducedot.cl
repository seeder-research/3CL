#define load_prod(i) (x1[i] * x2[i])

__kernel void
reducedot(__global float* __restrict x1, __global float* __restrict x2,
          volatile __global float* __restrict  dst, float initVal, int n) {
	reduce(load_prod, sum, atomic_add)
}

