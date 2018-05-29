// Kernel to convert a compact hermitian matrix into a full hermitian matrix
__kernel void hermitian2full(
   __global float2* dst,
   __global float2* src,
   const unsigned int sz,
   const unsigned int count)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access
	int init_offset = 1; // First entry that we need to copy

	__local float2 scratch[256];

	int currCnt = count; // Track how many items we have left to update in the array
	int currPivot = 128; // Default value of pivot. Will only get updated in last iteration
	while(global_idx < count) {
		float2 tmpR0 = src[global_idx + init_offset];
		if(currCnt < 256) {
			currPivot = currCnt / 2;
		}
		// Place reversed in local memory
		scratch[currPivot - lid + currPivot - 1] = tmpR0;
		barrier(CLK_LOCAL_MEM_FENCE);
		// Retrieve in order from local memory and store in global memory
		int outIdx = sz - global_idx;
		dst[outIdx] = scratch[local_idx];
		barrier(CLK_GLOBAL_MEM_FENCE);
		currCnt -= 256;
		global_idx += grp_offset;
	}
}
