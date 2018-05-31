// Kernel to convert a compact hermitian matrix into a full hermitian matrix
__kernel void
hermitian2full(__global float2* __restrict dst, __global float2* __restrict src, const unsigned int sz, const unsigned int count, __local float* scratchR, __local float* scratchI)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int currGrpOffset = grp_id * grp_sz;
	int global_idx = currGrpOffset + local_idx; // Calculate global index of work-item
	int ngrp = get_num_groups(0); // Get number of groups
	int grp_offset = ngrp * grp_sz; // Offset for memory access

	int currCnt = count - currGrpOffset; // Track how many items we have left to update in the array
	int currSize = (currCnt < grp_sz) ? currCnt : grp_sz; // Default value of group size.
	float2 tmpR0;

	while (currCnt > 0) {
		if (global_idx < count) {
			tmpR0 = src[global_idx]; // Grab src array element
			dst[global_idx] = tmpR0; // Copy the first half from src to dst
			tmpR0.y *= -1.0f;

			// Place reversed in local memory
			int tmp_idx = currSize - local_idx - 1;
			if (tmp_idx >= 0) {
				scratchR[tmp_idx] = tmpR0.x;
				scratchI[tmp_idx] = tmpR0.y;
			}
		}

		// Synchronize work-items
		barrier(CLK_LOCAL_MEM_FENCE);

		if (global_idx < count) {
			// Retrieve in order from local memory and store in global memory
			tmpR0.x = scratchR[local_idx];
			tmpR0.y = scratchI[local_idx];
			int outIdx = sz - currGrpOffset - currSize + local_idx + 1;
			if ((outIdx < sz) && (outIdx >= count)) {
				dst[outIdx] = tmpR0;
			}
		}

		// Synchronize work-items
		barrier(CLK_LOCAL_MEM_FENCE);

		grp_id += ngrp;
		global_idx += grp_offset;

		currGrpOffset = grp_id * grp_sz;
		currCnt = count - currGrpOffset; // Track how many items we have left to update in the array
		currSize = (currCnt < grp_sz) ? currCnt : grp_sz;
	}
}
