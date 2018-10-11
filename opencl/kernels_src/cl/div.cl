// dst[i] = a[i] / b[i]
__kernel void
pointwise_div(__global float* __restrict  dst, __global float* __restrict  a, __global float* __restrict b, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
        if (b[i] != 0.0f) {
            dst[i] = a[i] / b[i];
        } else {
            dst[i] = 0.0f;
        }
    }
}

