// C2C FFT kernel loop
__kernel void
ifft7_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns, int K_W) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	int sCount = 0;
	float angle_ = 2.0f * M_PI / (float)(K_W);

	while (sCount < Ns) {
		while (idx < N) {
			// Load input into registers
			float2 in0, in1, in2, in3, in4, in5, in6;
			int Idout0 = idx;
			int Idout1 = idx+N;
			int Idout2 = idx+2*N;
			int Idout3 = idx+3*N;
			int Idout4 = idx+4*N;
			int Idout5 = idx+5*N;
			int Idout6 = idx+6*N;

			in0 = dataIn[Idout0];
			in1 = dataIn[Idout1];
			in2 = dataIn[Idout2];
			in3 = dataIn[Idout3];
			in4 = dataIn[Idout4];
			in5 = dataIn[Idout5];
			in6 = dataIn[Idout6];
			in0 = fma(M_1_7, in0, 0.0f);
			in1 = fma(M_1_7, in1, 0.0f);
			in2 = fma(M_1_7, in2, 0.0f);
			in3 = fma(M_1_7, in3, 0.0f);
			in4 = fma(M_1_7, in4, 0.0f);
			in5 = fma(M_1_7, in5, 0.0f);
			in6 = fma(M_1_7, in6, 0.0f);

			// Perform length-7 inverse FFT calculations
			if (sCount > 0) {
				float angle = fma(angle_, (float)(idx), 0.0f);
				twiddle_factor(1, angle, in1);
				twiddle_factor(2, angle, in2);
				twiddle_factor(3, angle, in3);
				twiddle_factor(4, angle, in4);
				twiddle_factor(5, angle, in5);
				twiddle_factor(6, angle, in6);
			}

			IFFT7(in0, in1, in2, in3, in4, in5, in6);

			// Store results in place
			dataOut[Idout0] = in0;
			dataOut[Idout1] = in1;
			dataOut[Idout2] = in2;
			dataOut[Idout3] = in3;
			dataOut[Idout4] = in4;
			dataOut[Idout5] = in5;
			dataOut[Idout6] = in6;

			// Increment idx to go to next logical work item
			idx += inc;
			angle_ = fma(7.0f, angle_, 0.0f);
		}
		// TODO
		// Perform in-place matric transpose (treat matrix as N x 7 matrix and transpose)
		sCount++;
	}
}
