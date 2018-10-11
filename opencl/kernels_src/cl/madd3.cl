// dst[i] = fac1 * src1[i] + fac2 * src2[i] + fac3 * src3[i]
__kernel void
madd3(__global float* __restrict__ dst,
      __global float* __restrict src1, float fac1,
      __global float* __restrict src2, float fac2,
      __global float* __restrict src3, float fac3, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
	dst[i] = (fac1 * src1[i]) + (fac2 * src2[i] + fac3 * src3[i]);
	// parens for better accuracy heun solver.
    }
}

