// Landau-Lifshitz torque without precession
__kernel void
llnoprecess(__global float* __restrict  tx, __global float* __restrict  ty, __global float* __restrict  tz,
            __global float* __restrict  mx, __global float* __restrict  my, __global float* __restrict  mz,
            __global float* __restrict  hx, __global float* __restrict  hy, __global float* __restrict  hz, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

	float3 m = {mx[i], my[i], mz[i]};
	float3 H = {hx[i], hy[i], hz[i]};

	float3 mxH = cross(m, H);
	float3 torque = -cross(m, mxH);

	tx[i] = torque.x;
	ty[i] = torque.y;
	tz[i] = torque.z;
    }
}

