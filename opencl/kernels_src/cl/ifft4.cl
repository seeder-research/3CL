// C2C length 4 FFT kernel loop
__kernel void
ifft4_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns, int K_W) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	int sCount = 0;
	float angle_ = 2.0f * M_PI / (float)(K_W);

	while (sCount < Ns) {
		while (idx < N) {
			// Load input into registers
			float2 in0, in1, in2, in3;
			int Idout0 = idx;
			int Idout1 = idx+N;
			int Idout2 = idx+2*N;
			int Idout3 = idx+3*N;

			in0 = dataIn[Idout0];
			in1 = dataIn[Idout1];
			in2 = dataIn[Idout2];
			in3 = dataIn[Idout3];

			// Scale values
			in0 = fma(0.2500f, in0, 0.0f);
			in1 = fma(0.2500f, in1, 0.0f);
			in2 = fma(0.2500f, in2, 0.0f);
			in3 = fma(0.2500f, in3, 0.0f);

			// Perform length-4 inverse FFT calculations
			if (sCount > 0) {
				float angle = fma(angle_, (float)(idx), 0.0f);
				twiddle_factor(1, angle, in1);
				twiddle_factor(2, angle, in2);
				twiddle_factor(3, angle, in3);
			}

			IFFT4(in0, in1, in2, in3);

			// Store results in place
			dataOut[Idout0] = in0;
			dataOut[Idout1] = in1;
			dataOut[Idout2] = in2;
			dataOut[Idout3] = in3;

			// Increment idx to go to next logical work item
			idx += inc;
			angle_ = fma(4.0f, angle_, 0.0f);
		}
		// TODO
		// Perform in-place matric transpose (treat matrix as N x 4 matrix and transpose)
		sCount++;
	}
}
