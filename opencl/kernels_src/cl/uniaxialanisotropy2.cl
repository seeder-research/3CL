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

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

        float3 u   = normalized(vmul(ux_, uy_, uz_, ux_mul, uy_mul, uz_mul, i));
		float invMs = inv_Msat(Ms_, Ms_mul, i);
		float K1 = amul(K1_, K1_mul, i);
		float K2 = amul(K2_, K2_mul, i);
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

