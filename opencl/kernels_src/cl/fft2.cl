// C2C FFT kernel loop
__kernel void
fft2_c2c_long_interleaved_oop(__global float2* dataIn, __global float2* dataOut, int N, int Ns) {
	int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int inc = get_global_size(0);
	
	while (idx < N) {
		// Load input into registers
		float2 in0, in1, V;

		in0 = dataIn[idx];
		in1 = dataIn[idx+N];

		// Perform length-2 FFT calculations
		if (Ns != 1) {
			float angle = -2.0*M_PI*idx/N;
			twiddle_factor(1, angle, in1);
		}

		FFT2(in0, in1);

		// Store results
		int Idout = (idx/Ns)*Ns*K_W+(idx % Ns);
		dataOut[Idout] = in0;
		dataOut[Idout+Ns] = in1;

		// Increment idx to go to next logical work-item
		idx += inc;
	}
}
