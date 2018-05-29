// Kernel to convert a compact hermitian matrix into a full hermitian matrix
__kernel void hermitian2full(
   __global float2* dst,
   __global float2* src,
   __local float2* scratch,
   const unsigned int sz,
   const unsigned int count)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	int currCnt = count; // Track how many items we have left to update in the array
	int currSize = grp_sz; // Default value of pivot. Will only get updated in last iteration
	int outIdx = sz + local_idx;
	while(global_idx < count) {
		float2 tmpR0 = src[global_idx]; // Grab src array element
		dst[global_idx] = tmpR0; // Copy the first half from src to dst
		if (global_idx > 0) {
			tmpR0.y *= -1.0f;
			if(currCnt < grp_sz) {
				currSize = currCnt;
			}
			// Place reversed in local memory
			scratch[currSize - local_idx] = tmpR0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		// Retrieve in order from local memory and store in global memory
		outIdx -= currSize;
		if ((outIdx < sz) && (outIdx >= count)) {
			dst[outIdx] = scratch[local_idx];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		currCnt -= grp_sz;
		global_idx += grp_offset;
	}
}
