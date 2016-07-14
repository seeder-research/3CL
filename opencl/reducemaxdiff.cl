package kernels

var ReduceMaxDiffSource = `
#define load_diff(i) fabs(src1[i] - src2[i])

__kernel void
reducemaxdiff(__global float* __restrict src1, __global float* __restrict  src2, __global float* __restrict dst, float initVal, int n) {
	reduce(load_diff, fmax, atomicFmaxabs)
}

`
