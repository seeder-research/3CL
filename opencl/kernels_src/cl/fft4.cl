// C2C length 4 FFT kernel loop
__kernel void
fft4_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns) {
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
		if (Ns != 1) {
			float angle = -2.0*M_PI*(idx)/(N);
			twiddle_factor(1, angle, in1);
			twiddle_factor(2, angle, in2);
			twiddle_factor(3, angle, in3);
		}

		FFT$(in0, in1, in2, in3);

		// Store results to memory
		int Idout = (idx/Ns)*Ns*K_W+(idx % Ns);
		dataOut[Idout] = in0;
		dataOut[Idout+Ns] = in1;
		dataOut[Idout+2*Ns] = in2;
		dataOut[Idout+3*Ns] = in3;

		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
