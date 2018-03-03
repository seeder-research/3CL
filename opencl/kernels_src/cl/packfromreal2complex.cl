// Pack real matrix to complex matrix
__kernel void
packfromreal2complex(__global float* dataIn, __global float2* dataOut, int count) {
	int gid = get_global_id(0);
	if (gid < count) {
		float a = dataIn[gid];
		float2 b;
		b = (float2){a, 0.0f};
		dataOut[gid] = b;
	}
}
