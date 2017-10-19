// C2C length 8 FFT kernel loop
__kernel void
fft8_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	
	while (idx < N) {
		// Load input into registers
		float2 in0, in1, in2, in3, in4, in5, in6, in7;
		in0 = dataIn[idx];
		in1 = dataIn[idx+N];
		in2 = dataIn[idx+2*N];
		in3 = dataIn[idx+3*N];
		in4 = dataIn[idx+4*N];
		in5 = dataIn[idx+5*N];
		in6 = dataIn[idx+6*N];
		in7 = dataIn[idx+7*N];

		// Perform FFT computations using Cooley-Tukey
		float2 v0, v1, v2, v3, v4, v5, v6, v7;
		//		FFT4(in0,in2,in4,in6)
		//		FFT4(in1,in3,in5,in7)

		// First FFT4()
		v0 = in0 + in4;
		v1 = in2 + in6;
		v2 = in0 - in4;
		v3.x = in2.y - in6.y;
		v3.y = in2.x - in6.x;
		in0 = v0 + v1;
		in4 = v0 - v1;
		in2 = v2 + v3;
		in6 = v2 - v3;

		// Second FFT4()
		v4 = in1 + in5;
		v5 = in3 + in7;
		v6 = in1 - in5;
		v7.x = in3.y - in7.y;
		v7.y = in3.x - in7.x;
		in1 = v4 + v5;
		in5 = v4 - v5;
		in3 = v6 + v7;
		in7 = v6 - v7;
		
		// Transfer results of first FFT4()
		v0 = in0;
		v2 = in2;
		v4 = in4;
		v6 = in6;

		// Transfer results of second FFT4() with twiddle multiplication
		float angle = M_PI/4;
		v1 = in1;
		v3 = twiddle(1, angle, in3);
		v5 = twiddle(2, angle, in5);
		v7 = twiddle(3, angle, in7);

		in0 = v0 + v1;
		in1 = v0 - v1;
		in2 = v2 + v3;
		in3 = v2 - v3;
		in4 = v4 + v5;
		in5 = v4 - v5;
		in6 = v6 + v7;
		in7 = v6 - v7;
		
		// Store results to memory
		dataOut[idx] = in0;
		dataOut[idx+N] = in1;
		dataOut[idx+2*N] = in2;
		dataOut[idx+3*N] = in3;
		dataOut[idx+4*N] = in4;
		dataOut[idx+5*N] = in5;
		dataOut[idx+6*N] = in6;
		dataOut[idx+7*N] = in7;

		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
