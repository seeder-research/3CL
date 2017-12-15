// C2C FFT kernel loop
__kernel void
ifft2_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns, int K_W) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	int sCount = 0;
	float angle_ = 2.0f * M_PI / (float)(K_W);

	while (sCount < Ns) {
		while (idx < N) {
			// Load input into registers
			float2 in0, in1, V;
			int Idout0 = idx;
			int Idout1 = Idout0+N;

			in0 = dataIn[Idout0];
			in1 = dataIn[Idout1];

			// Scale values
			in0 *= 0.5000000000f;
			in1 *= 0.5000000000f;

			// Perform length-2 inverse FFT calculations
			if (sCount > 0) {
				float angle = angle_ * (float)(idx);
				twiddle_factor(1, angle, in1);
			}

			FFT2(in0, in1); // Same whether we do forward or inverse

			// Store results in place
			dataOut[Idout0] = in0;
			dataOut[Idout1] = in1;

			// Increment idx to go to next logical work item
			idx += inc;
			angle_ *= 2.0f;
		}
		// TODO
		// Perform in-place matric transpose (treat matrix as N x 2 matrix and transpose)
		sCount++;
	}
}
