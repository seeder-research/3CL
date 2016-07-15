#define load_vecdiff2(i)  \
	pow2(x1[i] - x2[i]) + \
	pow2(y1[i] - y2[i]) + \
	pow2(z1[i] - z2[i])   \
 
__kernel void
reducemaxvecdiff2(__global float* __restrict x1, __global float* __restrict y1, __global float* __restrict z1,
                  __global float* __restrict x2, __global float* __restrict y2, __global float* __restrict z2,
                  __global float* __restrict dst, float initVal, int n) {
	reduce(load_vecdiff2, fmax, atomicFmaxabs)
}

