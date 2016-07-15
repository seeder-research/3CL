#define load(i) src[i]

__kernel void
reducesum(__global float* __restrict src, __global float*__restrict  dst, float initVal, int n) {
	reduce(load, sum, atomic_add)
}

