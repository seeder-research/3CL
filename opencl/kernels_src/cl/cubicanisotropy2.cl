// add cubic anisotropy field to B.
// B:      effective field in T
// m:      reduced magnetization (unit length)
// Ms:     saturation magnetization in A/m.
// K1:     Kc1 in J/m3
// K2:     Kc2 in T/m3
// C1, C2: anisotropy axes
//
// based on http://www.southampton.ac.uk/~fangohr/software/oxs_cubic8.html
__kernel void
addcubicanisotropy2(__global float* __restrict Bx, __global float* __restrict By, __global float* __restrict Bz,
                    __global float* __restrict mx, __global float* __restrict my, __global float* __restrict mz,
                    __global float* __restrict  Ms_, float  Ms_mul,
                    __global float* __restrict  k1_, float  k1_mul,
                    __global float* __restrict  k2_, float  k2_mul,
                    __global float* __restrict  k3_, float  k3_mul,
                    __global float* __restrict c1x_, float c1x_mul,
                    __global float* __restrict c1y_, float c1y_mul,
                    __global float* __restrict c1z_, float c1z_mul,
                    __global float* __restrict c2x_, float c2x_mul,
                    __global float* __restrict c2y_, float c2y_mul,
                    __global float* __restrict c2z_, float c2z_mul,
                    int N) {

    int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
    if (i < N) {

        float ms = (Ms_ == NULL) ? (Ms_mul) : (Ms_mul * Ms_[i]);
        float invMs = (ms == 0.0f) ? (ms) : (1.0f / ms);
        float  k1 = (k1_ == NULL) ? k1_mul : (k1_mul * k1_[i]);
	k1 *= invMs;
        float  k2 = (k2_ == NULL) ? k2_mul : (k2_mul * k2_[i]);
	k2 *= invMs;
        float  k3 = (k3_ == NULL) ? k3_mul : (k3_mul * k3_[i]);
	k3 *= invMs;
	float u1x = (c1x_ == NULL) ? c1x_mul : (c1x_mul * c1x_[i]);
	float u1y = (c1y_ == NULL) ? c1y_mul : (c1y_mul * c1y_[i]);
	float u1z = (c1z_ == NULL) ? c1z_mul : (c1z_mul * c1z_[i]);
        float3 u1 = normalized(make_float3(u1x, u1y, u1z));
	float u2x = (c2x_ == NULL) ? c2x_mul : (c2x_mul * c2x_[i]);
	float u2y = (c2y_ == NULL) ? c2y_mul : (c2y_mul * c2y_[i]);
	float u2z = (c2z_ == NULL) ? c2z_mul : (c2z_mul * c2z_[i]);
        float3 u2 = normalized(make_float3(u2x, u2y, u2z));
        float3 u3 = cross(u1, u2); // 3rd axis perpendicular to u1,u2
        float3 m  = make_float3(mx[i], my[i], mz[i]);

        float u1m = dot(u1, m);
        float u2m = dot(u2, m);
        float u3m = dot(u3, m);

        float3 B = -2.0f*k1*((pow2(u2m) + pow2(u3m)) * (    (u1m) * u1) +
                             (pow2(u1m) + pow2(u3m)) * (    (u2m) * u2) +
                             (pow2(u1m) + pow2(u2m)) * (    (u3m) * u3))-
                   2.0f*k2*((pow2(u2m) * pow2(u3m)) * (    (u1m) * u1) +
                            (pow2(u1m) * pow2(u3m)) * (    (u2m) * u2) +
                            (pow2(u1m) * pow2(u2m)) * (    (u3m) * u3))-
                   4.0f*k3*((pow4(u2m) + pow4(u3m)) * (pow3(u1m) * u1) +
                            (pow4(u1m) + pow4(u3m)) * (pow3(u2m) * u2) +
                            (pow4(u1m) + pow4(u2m)) * (pow3(u3m) * u3));
        Bx[i] += B.x;
        By[i] += B.y;
        Bz[i] += B.z;
    }
}
