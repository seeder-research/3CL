#define load_vecnorm2(i) \
	pow2(x[i]) + pow2(y[i]) +  pow2(z[i])

__kernel void
reducemaxvecnorm2(__global float* __restrict x, __global float* __restrict y, __global float* __restrict z, __global float* __restrict dst, float initVal, int n) {
	reduce(load_vecnorm2, fmax, atomicFmaxabs)
}

