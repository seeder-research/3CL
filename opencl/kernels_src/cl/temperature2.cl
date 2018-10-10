// TODO: this could act on x,y,z, so that we need to call it only once.
__kernel void
settemperature2(__global float* __restrict  B,      __global float* __restrict noise, float kB2_VgammaDt,
                __global float* __restrict Ms_, float Ms_mul,
                __global float* __restrict temp_, float temp_mul,
                __global float* __restrict alpha_, float alpha_mul,
                int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float temp = amul(temp_, temp_mul, i);
        float alpha = amul(alpha_, alpha_mul, i);
        B[i] = noise[i] * sqrt((kB2_VgammaDt * alpha * temp * invMs ));
    }
}

