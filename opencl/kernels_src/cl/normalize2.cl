// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
__kernel void
normalize2(__global float* __restrict vx, __global float* __restrict vy, __global float* __restrict vz, __global float* __restrict vol, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

	float v = (vol == NULL) ? 1.0f : vol[i];
	float3 V = {v*vx[i], v*vy[i], v*vz[i]};
	V = normalize(V);
	if (v == 0.0f) {
		vx[i] = 0.0;
		vy[i] = 0.0;
		vz[i] = 0.0;
	} else {
		vx[i] = V.x;
		vy[i] = V.y;
		vz[i] = V.z;
	}
    }
}

