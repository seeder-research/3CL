// Complex multiplication
// dst[i] = (a[i].x + i*a[i].y)*(b[i].x + i*b[i].y)
// dst[i].x = a[i].x*b[i].x - a[i].y*b[i].y
// dst[i].y = a[i].x*b[i].y + a[i].y*b[i].x
__kernel void
cmplx_mul(__global float2* __restrict  dst, __global float2* __restrict  a, __global float2* __restrict b, const unsigned int conjB, const unsigned int N, const unsigned int offset) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
	float2 aa = a[i+offset];
	float2 bb = b[i+offset];
	float xx, yy, xy, yx;
	float2 dd;
	xx = aa.x * bb.x;
	yy = aa.y * bb.y;
	xy = aa.x * bb.y;
	yx = aa.y * bb.x;
	if (conjB == 0) {
		dd.x = xx - yy;
		dd.y = xy + yx;
	} else {
		dd.x = xx + yy;
		dd.y = yx - xy;
	}
	dst[i+offset] = dd;
    }
}
