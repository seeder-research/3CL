// add region-based vector to dst:
// dst[i] += LUT[region[i]]
__kernel void
regionaddv(__global float* __restrict dstx, __global float* __restrict dsty, __global float* __restrict dstz,
           __global float* __restrict LUTx, __global float* __restrict LUTy, __global float* __restrict LUTz,
           __global uint8_t* regions, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

	uint8_t r = regions[i];
	dstx[i] += LUTx[r];
	dsty[i] += LUTy[r];
	dstz[i] += LUTz[r];
    }
}

