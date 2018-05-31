// Complex multiplication
// dst[i] = a[i] * b[i]
__kernel void
cmplx_mul(__global float2* __restrict  dst, __global float2* __restrict  a, __global float2* __restrict b, const unsigned int N, const unsigned int offset) {

	int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);

	if(i < N) {
		float2 aa = a[i+offset];
		float2 bb = b[i+offset];
		float2 cc;
		cc.x = aa.x * bb.x - (aa.y * bb.y);
		cc.y = aa.x * bb.y + bb.x * aa.y;
		dst[i+offset] = cc;
	}
}
