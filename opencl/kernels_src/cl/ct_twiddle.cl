// C2C length 8 FFT kernel loop
__kernel void
ct_twiddle_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, float angle, int Cnt) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	
	while (idx < N) {
		// Initialize registers
		float2 in0, v0;
		
		// Initialize position pointer
		int pos = 1;

		// Iterate through array and multiply twiddle factor
		while (pos < Cnt) {
			in0 = dataIn[idx+pos*N];
			v0 = twiddle(pos*idx, angle, in0);
			dataOut[idx+pos*N] = v0;
		}
	
		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
