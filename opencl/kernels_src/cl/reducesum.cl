__kernel void
reducesum(__global float* __restrict src, __global float* __restrict dst, float initVal, int n, __local float* scratch) {
	// Initialize memory
	int global_idx =  get_global_id(0);
	int local_idx = get_local_id(0);
	float currVal = 0;
	float residualAcc = 0;
	float yTmp = 0;
	float tTmp = 0;
	
	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		currVal = initVal;
	}

	// Loop over input elements in chunks and accumulate each chunk into local memory
	while (global_idx < n) {
		yTmp = src[global_idx] - residualAcc;
		tTmp = currVal + yTmp;
		residualAcc = (tTmp - currVal) - yTmp;
		currVal = tTmp;
		global_idx += get_global_size(0);
	}

	// At this point, accumulated values on chunks are in local memory. Perform parallel reduction
	scratch[local_idx] = currVal;
	residualAcc = 0;
	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if (local_idx < offset) {
			yTmp = scratch[local_idx + offset] - residualAcc;
			tTmp = scratch[local_idx] + yTmp;
			residualAcc = (tTmp - scratch[local_idx]) - yTmp;
			scratch[local_idx] = tTmp;
		}
		// barrier for syncing work group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_idx == 0) {
		dst[get_group_id(0)] = scratch[0];
	}
}

