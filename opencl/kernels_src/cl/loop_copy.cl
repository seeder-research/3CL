// Test of kernel loop
__kernel void
loop_copy(__global float* dst, __global float* src, int N) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	
	while (idx < N) {
		// Sequence of instructions to execute
		dst[idx] = src[idx];
		
		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
