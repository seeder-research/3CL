// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat * vol
__kernel void
copypadmul2(__global float* __restrict dst, int Dx, int Dy, int Dz,
            __global float* __restrict src, int Sx, int Sy, int Sz,
            __global float* __restrict Ms_, float Ms_mul,
            __global float* __restrict vol) {

    int ix = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int iy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int iz = get_group_id(2) * get_local_size(2) + get_local_id(2);

    if (ix<Sx && iy<Sy && iz<Sz) {
        int sI = index(ix, iy, iz, Sx, Sy, Sz);  // source index
	float tmpFac = (Ms_ == NULL) ? (Ms_mul) : (Ms_mul * Ms_[sI]);
        float Bsat = MU0 * tmpFac;
        float v = (vol == NULL) ? (1.0f) : (1.0f * vol[sI]);
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = Bsat * v * src[sI];
    }
}

