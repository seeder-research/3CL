__kernel void
reducesum(__global float* __restrict src, __global float* __restrict dst, float initVal, int n, __local float* scratch1, __local float* scratch2) {
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	// Initialize memory
	float currVal = 0;
	float localErr = 0;
	scratch2[local_idx] = 0;
	float2 y;
	float2 t;
	float2 u;
	float v;
	
	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		currVal = initVal;
	}

	// Loop over input elements in chunks and accumulate each chunk into local memory
	while (global_idx < n) {
		y.x = src[global_idx];
		y.y = currVal;
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		currVal = u.x;
		y = u - y;
		u = y - t;
		v = u.x + u.y;
		localErr += v;
		global_idx += grp_offset;
	}

	// At this point, accumulated values on chunks are in local memory. Perform parallel reduction
	scratch1[local_idx] = currVal;
	scratch2[local_idx] = localErr;
	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = grp_sz / 2; offset > 32; offset >>= 1) {
		y.x = scratch1[local_idx + offset];
		y.y = scratch1[local_idx];
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		scratch1[local_idx] = u.x;
		y = u - y;
		u = y - t;
		localErr = u.x + u.y;
		y.x = scratch2[local_idx + offset];
		y.y = scratch2[local_idx];
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		scratch2[local_idx] = u.x;
		y = u - y;
		u = y - t;
		v = u.x + u.y;
		scratch2[local_idx] += localErr - v;
		// barrier for syncing work group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx == 0) {
		dst[grp_id] = scratch1[0] - scratch2[0];
	}
}
