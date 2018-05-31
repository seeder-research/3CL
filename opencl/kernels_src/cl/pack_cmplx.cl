// Kernel to transfer an array of reals to a complex array
__kernel void
pack_cmplx(__global float2* dst, __global float* src, const unsigned int count)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	while(global_idx < count) {
		float a0 = src[global_idx];
		float a1 = 0.0f;
		float2 b0;
		b0.x = a0;
		b0.y = a1;
		dst[global_idx] = b0;
		global_idx += grp_offset;
	}
}
