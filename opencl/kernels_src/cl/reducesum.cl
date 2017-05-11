__kernel void
reducesum(__global float* __restrict src, __global float* __restrict dst, float initVal, int n, __local float* scratch1, __local float* scratch2) {
	// Initialize memory
	int global_idx =  get_global_id(0);
	int local_idx = get_local_id(0);
	float currVal = 0;
	scratch2[local_idx] = 0;
	float yTmp = 0;
	float tTmp = 0;
	
	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		currVal = initVal;
	}

	// Loop over input elements in chunks and accumulate each chunk into local memory
	while (global_idx < n) {
		float nextVal = src[global_idx];
		if (fabs(tTmp) > fabs(nextVal)) {
			yTmp = nextVal - scratch2[local_idx];
			tTmp = currVal + yTmp;
			scratch2[local_idx] = (tTmp - currVal) - yTmp;
			currVal = tTmp;
		} else {
			yTmp = currVal - scratch2[local_idx];
			tTmp = nextVal + yTmp;
			scratch2[local_idx] = (tTmp - nextVal) - yTmp;
			currVal = tTmp;
		}
		global_idx += get_global_size(0);
	}

	// At this point, accumulated values on chunks are in local memory. Perform parallel reduction
	scratch1[local_idx] = currVal;
	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if (local_idx < offset) {
			if (fabs(scratch1[local_idx]) > fabs(scratch1[local_idx + offset])) {
				yTmp = (scratch1[local_idx + offset] - scratch2[local_idx]) - scratch2[local_idx + offset];
				tTmp = scratch1[local_idx] + yTmp;
				scratch2[local_idx] = (tTmp - scratch1[local_idx]) - yTmp;
				scratch1[local_idx] = tTmp;
			} else {
				yTmp = (scratch1[local_idx] - scratch2[local_idx]) - scratch2[local_idx + offset];
				tTmp = scratch1[local_idx + offset] + yTmp;
				scratch2[local_idx] = (tTmp - scratch1[local_idx + offset]) - yTmp;
				scratch1[local_idx] = tTmp;
			}
		}
		// barrier for syncing work group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx == 0) {
		dst[get_group_id(0)] = scratch1[0];
	}
}

