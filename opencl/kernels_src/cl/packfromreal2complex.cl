// Pack real matrix to complex matrix
__kernel void
packfromreal2complex(__global float* dataIn, __global float* dataOut, int count) {
	int gid = get_global_id(0);
	if (gid < count) {
		float a = dataIn[gid];
		dataOut[2*gid] = a;
		dataOut[2*gid + 1] = 0.0f;
	}
}
