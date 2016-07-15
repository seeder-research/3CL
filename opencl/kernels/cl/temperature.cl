__kernel void
settemperature(__global float* __restrict  B,      __global float* __restrict noise, float kB2_VgammaDt,
               __global float* __restrict tempRedLUT, __global uint8_t* __restrict regions, int N) {

	int i =   ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
	if (i < N) {
		uint8_t reg  = regions[i];
		float alphaT_Bs  = tempRedLUT[reg];
		B[i] = noise[i] * native_sqrt( kB2_VgammaDt * alphaT_Bs );
	}
}

