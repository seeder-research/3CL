// dst += prefactor * dot(a,b)
__kernel void
dotproduct(__global float* __restrict  dst, float  prefactor,
           __global float* __restrict  ax, __global float* __restrict  ay, __global float* __restrict  az,
           __global float* __restrict  bx, __global float* __restrict  by, __global float* __restrict  bz,
           int  N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
	float3 A = {ax[i], ay[i], az[i]};
	float3 B = {bx[i], by[i], bz[i]};
	dst[i] += prefactor * dot(A, B);
    }
}

