package kernels

var UniaxialAnisotropySource = `
__kernel void
adduniaxialanisotropy(__global float* __restrict  Bx, __global float* __restrict  By, __global float* __restrict  Bz,
                      __global float* __restrict  mx, __global float* __restrict  my, __global float* __restrict  mz,
                      __global float* __restrict K1LUT, __global float* __restrict K2LUT,
                      __global float* __restrict uxLUT, __global float* __restrict uyLUT, __global float* __restrict uzLUT,
                      __global uint8_t* __restrict regions, int N) {

	int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
	if (i < N) {

		uint8_t reg = regions[i];
		float  ux  = uxLUT[reg];
		float  uy  = uyLUT[reg];
		float  uz  = uzLUT[reg];
		float3 u   = normalized(make_float3(ux, uy, uz));
		float  K1  = K1LUT[reg];
		float  K2  = K2LUT[reg];
		float3 m   = {mx[i], my[i], mz[i]};
		float  mu  = dot(m, u);
		float3 Ba  = 2.0f*K1*    (mu)*u+ 
                     4.0f*K2*pow3(mu)*u;

		Bx[i] += Ba.x;
		By[i] += Ba.y;
		Bz[i] += Ba.z;
	}
}

`
