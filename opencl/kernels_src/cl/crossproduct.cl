__kernel void
crossproduct(__global float* __restrict  dstx, __global float* __restrict  dsty, __global float* __restrict  dstz,
           __global float* __restrict ax, __global float* __restrict ay, __global float* __restrict az,
           __global float* __restrict bx, __global float* __restrict by, __global float* __restrict bz,
           int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
	float3 AxB = cross(A, B);
        dstx[i] = AxB.x;
        dsty[i] = AxB.y;
        dstz[i] = AxB.z;
    }
}

