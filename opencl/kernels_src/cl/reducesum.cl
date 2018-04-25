__kernel void
reducesum(__global float* __restrict src, __global float* __restrict dst, float initVal, int n, __local float* scratch1, __local float* scratch2) {
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	// Initialize registers for work-item
	float2 y; // Temporary register
	float2 t; // Temporary register
	float2 u; // Temporary register
	float v; // Temporary register
	float lsum = 0; // Register to temporarily store A + B
	float lerr = 0; // Register to temporarily store error from A + B
	float lerr2 = 0; // Register to temporarily store error from summing errors
	
	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		lsum = initVal;
	}

/*
	// During each loop iteration, we:
	// 1) add source data into local accumulator. If global index exceeds the position, then we load 0
	// 2) once all source data has been scanned, load accumulator and error values into scratch1 and scratch2
	// 3) perform a reduction sum over the values in scratch1 with the errors in scratch2
	// 4) Accumulate into accumulator in work-item with local_idx = 0
*/

	while (global_idx < n) {
		y.x = src[global_idx];
		y.y = lsum;
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		lsum = u.x;
		y = u - y;
		u = y - t;
		v = u.x + u.y;
		y.x = v;
		y.y = lerr;
		t.x = y.y;
		t.y = y.x;
		u = y + t;
		lerr = u.x;
		y = u - y;
		u = y - t;
		lerr2 += u.x + u.y;
		global_idx += grp_offset;
	}

	scratch1[local_idx] = lsum;
	scratch2[local_idx] = lerr - lerr2;

	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = grp_sz / 2; offset > 32; offset >>= 1) {
		if (local_idx < offset) {
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
		}
		// barrier for syncing workgroup
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx < 32) {
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 32];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 32];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 16];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 16];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 8];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 8];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 4];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 4];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 2];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 2];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
			y.x = scratch1[local_idx];
			y.y = scratch1[local_idx + 1];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lsum = u.x;
			y = u - y;
			u = y - t;
			lerr2 = u.x + u.y;
			y.x = scratch2[local_idx];
			y.y = scratch2[local_idx + 1];
			barrier(CLK_LOCAL_MEM_FENCE);
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			v = u.x + u.y;
			y.x = lerr;
			y.y = lerr2;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr = u.x;
			y = u - y;
			u = y - t;
			y.x = u.x + u.y;
			y.y = v;
			t.x = y.y;
			t.y = y.x;
			u = y + t;
			lerr2 = u.x;
			scratch1[local_idx] = lsum;
			scratch2[local_idx] = lerr - lerr2;
			barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx == 0) {
		dst[grp_id] = scratch1[0] - scratch2[0];
	}
}
