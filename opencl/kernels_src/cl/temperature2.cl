// TODO: this could act on x,y,z, so that we need to call it only once.
__kernel void
settemperature2(__global float* __restrict  B,      __global float* __restrict noise, float kB2_VgammaDt,
                __global float* __restrict Ms_, float Ms_mul,
                __global float* __restrict temp_, float temp_mul,
                __global float* __restrict alpha_, float alpha_mul,
                int N) {

    int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
    if (i < N) {
		float msat = (Ms_ == NULL) ? (Ms_mul) : (Ms_mul * Ms_[i]);
        float invMs = (msat == 0.0f) ? (0.0f) : (1.0f / msat);
        float temp = (temp_ == NULL) ? (temp_mul) : (temp_mul * temp_[i]);
        float alpha = (alpha_ == NULL) ? (alpha_mul) : (alpha_mul * alpha_[i]);
        B[i] = noise[i] * sqrt((kB2_VgammaDt * alpha * temp * invMs ));
    }
}

