__kernel void
reducedot(__global float* __restrict x1, __global float* __restrict x2,
          volatile __global float* __restrict  dst, float initVal, int n, __local float* scratch1, __local float* scratch2) {

	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	// Initialize memory
	float2 y;
	float2 t;
	float2 u;
	float tmpR0;
	float currVal = 0;
	float currErr = 0;

	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		currVal = initVal;
	}

	// Loop over input elements in chunks and accumulate each chunk into local memory
	while (global_idx < n) {
		tmpR0 = x1[global_idx] * x2[global_idx];
		y.x = currVal;
		y.y = tmpR0;
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		currVal = u.x;
		y = u - y;
		u = y - t;
		currErr += u.x + u.y;
		global_idx += grp_offset;
	}

	// At this point, accumulated values on chunks are in local memory. Perform parallel reduction
	scratch1[local_idx] = currVal;
	scratch2[local_idx] = currErr;

	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = grp_sz / 2; offset > 32; offset >>= 1) {
		if (local_idx < offset) {
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
		}
		// barrier for syncing work group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx < 32) {
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 32];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 32];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 16];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 16];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 8];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 8];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 4];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 4];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 2];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 2];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 1];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			currErr = scratch2[local_idx] + scratch2[local_idx + 1];
			barrier(CLK_LOCAL_MEM_FENCE);
			u = y + t;
			currVal = u.x;
			y = u - y;
			u = y - t;
			currErr += u.x + u.y;
			scratch1[local_idx] = currVal;
			scratch2[local_idx] = currErr;
			barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_idx == 0) {
		dst[grp_id] = scratch1[0] - scratch2[0];
	}
}
