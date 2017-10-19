// C2C length 4 FFT kernel loop
__kernel void
fft4_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	
	while (idx < N) {
		// Load input into registers
		float2 in0, in1, in2, in3;
		in0 = dataIn[idx];
		in1 = dataIn[idx+N];
		in2 = dataIn[idx+2*N];
		in3 = dataIn[idx+3*N];

		// Perform FFT computations
		float2 v0, v1, v2, v3;
		v0 = in0 + in2;
		v1 = in1 + in3;
		v2 = in0 - in2;
		v3.x = in1.y - in3.y;
		v3.y = in1.x - in3.x;
		in0 = v0 + v1;
		in2 = v0 - v1;
		in1 = v2 + v3;
		in3 = v2 - v3;

		// Store results to memory
		dataOut[idx] = in0;
		dataOut[idx+N] = in1;
		dataOut[idx+2*N] = in2;
		dataOut[idx+3*N] = in3;

		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
