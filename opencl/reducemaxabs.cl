package kernels

var ReduceMaxAbsSource = `
#define load_fabs(i) fabs(src[i])

__kernel void
reducemaxabs(__global float* __restrict src, __global float* __restrict dst, float initVal, int n) {
	reduce(load_fabs, fmax, atomicFmaxabs)
}

`
