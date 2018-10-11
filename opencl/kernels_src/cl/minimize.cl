// Steepest descent energy minimizer
__kernel void
minimize(__global float* __restrict mx,  __global float* __restrict  my,  __global float* __restrict mz,
         __global float* __restrict m0x, __global float* __restrict  m0y, __global float* __restrict m0z,
         __global float* __restrict tx,  __global float* __restrict  ty,  __global float* __restrict tz,
         float dt, int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

	float3 m0 = {m0x[i], m0y[i], m0z[i]};
	float3 t = {tx[i], ty[i], tz[i]};

	float t2 = dt*dt*dot(t, t);
	float3 result = (4 - t2) * m0 + 4 * dt * t;
	float divisor = 4 + t2;
		
	mx[i] = result.x / divisor;
	my[i] = result.y / divisor;
	mz[i] = result.z / divisor;
    }
}

