// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
__kernel void
adduniaxialanisotropy2(__global float* __restrict  Bx, __global float* __restrict  By, __global float* __restrict  Bz,
                       __global float* __restrict  mx, __global float* __restrict  my, __global float* __restrict  mz,
                       __global float* __restrict Ms_, float Ms_mul,
                       __global float* __restrict K1_, float K1_mul,
                       __global float* __restrict K2_, float K2_mul,
                       __global float* __restrict ux_, float ux_mul,
                       __global float* __restrict uy_, float uy_mul,
                       __global float* __restrict uz_, float uz_mul,
                       int N) {

    int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
    if (i < N) {

	float ux = (ux_ == NULL) ? (ux_mul) : (ux_mul * ux_[i]);
	float uy = (uy_ == NULL) ? (uy_mul) : (uy_mul * uy_[i]);
	float uz = (uz_ == NULL) ? (uz_mul) : (uz_mul * uz_[i]);
        float3 u   = normalized(make_float3(ux, uy, uz));
        float ms = (Ms_ == NULL) ? (Ms_mul) : (Ms_mul * Ms_[i]);
	float invMs = (ms == 0.0f) ? (ms) : (1.0f / ms);
	float K1 = (K1_ == NULL) ? (K1_mul) : (K1_mul * K1_[i]);
	float K2 = (K2_ == NULL) ? (K2_mul) : (K2_mul * K2_[i]);
        K1  *= invMs;
        K2  *= invMs;
        float3 m   = {mx[i], my[i], mz[i]};
        float  mu  = dot(m, u);
        float3 Ba  = 2.0f*K1*    (mu)*u+
                     4.0f*K2*pow3(mu)*u;

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}

