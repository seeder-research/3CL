__kernel void
reducesum(__global float* __restrict src, __global float* __restrict dst, float initVal, int n, __local float* scratch1, __local float* scratch2) {
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	// Initialize registers for work-item
	float currSum = 0; // Local accumulator
	float currErr = 0; // Track error for accumulator
	float tmpR0 = 0; // Temporary register
	float tmpR1 = 0; // Temporary register
	float tmpR2 = 0; // Temporary register
	float tmpR3 = 0; // Temporary register
	float tmpR4 = 0; // Temporary register
	
	// Set the accumulator value to initVal for the first work-item only
	if (global_idx == 0) {
		currSum = initVal;
	}

	// Loop over input elements in chunks and accumulate each chunk into local memory
	while (global_idx < n) {
		tmpR0 = src[global_idx]; // Load next number into local register
		tmpR1 = fma(1.0f, currSum, tmpR0); // Temporary sum

		// Calculate the operands of the sum from temporary sum
		tmpR2 = fma(-1.0f, tmpR0, tmpR1); // Recovers currSum with error
		tmpR3 = fma(-1.0f, currSum, tmpR1); // Recovers next value with error
		tmpR4 = fma(-1.0f, currSum, tmpR2);; // Error in currSum
		tmpR2 = fma(-1.0f, tmpR0, tmpR3); // Error in next value
		currSum = tmpR1; // Update sum into accumulator
		currErr = fma(1.0f, currErr, fma(1.0f, tmpR4, tmpR2)); // Accumulate errors
		global_idx += grp_offset;
	}

	// At this point, accumulated values on chunks are in local memory. Perform parallel reduction
	scratch1[local_idx] = currSum;
	scratch2[local_idx] = currErr;

	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if (local_idx < offset) {
			currSum = scratch1[local_idx]; // Load accumulator
			currErr = fma(1.0f, scratch2[local_idx], scratch2[local_idx + offset]); // Load and accumulate error
			tmpR0 = scratch1[local_idx + offset]; // Load next number into local register
			tmpR1 = fma(1.0f, currSum, tmpR0); // Temporary sum

			// Calculate the operands of the sum from temporary sum
			tmpR2 = fma( -1.0f, tmpR0, tmpR1); // Recovers currSum with error
			tmpR3 = fma( -1.0f, currSum, tmpR1); // Recovers next value with error
			tmpR4 = fma( -1.0f, currSum, tmpR2); // Error in currSum
			tmpR2 = fma( -1.0f, tmpR0, tmpR3); // Error in next value
			currSum = tmpR1; // Store into accumulator
			currErr = fma(1.0f, currErr, fma(1.0f, tmpR4, tmpR2)); // Accumulate errors

			// Write results back into workgroup scratch memory
			scratch1[local_idx] = currSum;
			scratch2[local_idx] = currErr;
		}
		// barrier for syncing workgroup
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Add barrier to sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	if (local_idx == 0) {
		dst[grp_id] = fma(-1.0f, scratch2[0], scratch1[0]);
	}
}

