// C2C FFT kernel loop
// Strategy:
// 1. Assume input is in order and we are doing N number of FFT2
//    on the entire length of the input (i.e. input has 2*N elements)
// 2. Each kernel is doing a separate FFT in the batch
// 3. Kernel should perform FFT2 for a number of times defined
//    by input
// 4. First N elements will be first input to each FFT2, next N
//    elements are the second input to each FFT2
// 5. Check flag (Ns) to see which stages we are in.
//    If we are in the first stage, we do not need to
//    multiply twiddle factors prior to FFT. If not,
//    we need to multiply by twiddle factors first before proceeding
// 6. Use flag to decide whether we should transpose result. If we
//    need to, we should compute output index. Otherwise, we just
//    write back to same index as where we got input.
__kernel void
fft2_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns, int K_W) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	int sCount = 0;
	float angle_ = -2.0f * M_PI / (float)(K_W);

	while (sCount < Ns) {
		while (idx < N) {
			// Load input into registers
			float2 in0, in1, V;
			int Idout0 = idx;
			int Idout1 = Idout0+N;

			in0 = dataIn[Idout0];
			in1 = dataIn[Idout1];

			// Perform length-2 forward FFT calculations
			if (sCount > 0) {
				float angle = angle_ * (float)(idx);
				twiddle_factor(1, angle, in1);
			}

			FFT2(in0, in1);

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

/*
__kernel void
fft2_c2c_adj_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns, int Np) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	int sCount = 0;
	float angle_ = -2.0*M_PI/K_W;
	__local float2 [32]cache;
	int	lidx = get_local_id(0)*2;
	
	while (sCount < Ns) {
		while (idx < N) {
			// Load input into registers
			float2 in0, in1, V;

			in0 = dataIn[idx]; // Load from main memory into registers
			cache[get_local_id(0)] = in0;

			// Perform length-2 FFT calculations
			if (sCount > 0) {
				float angle = angle_*idx;
				twiddle_factor(1, angle, in1);
			}

			FFT2(in0, in1);

			// Store results
			// Computing where to store results depending on whether to transpose
			int Idout  = idx;
			int Idout_ = idx+N;

			if (Ns > 1) {
				Idout  = 2*N;
				Idout_ = Idout+1;
			}

			dataOut[Idout] = in0;
			dataOut[Idout_] = in1;

			// Increment idx to go to next logical work item
			idx += inc;
			angle_ *= 2;
		}
		sCount++;
	}
}
*/
