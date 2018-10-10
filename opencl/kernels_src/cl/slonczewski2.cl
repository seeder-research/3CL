// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016

__kernel void
addslonczewskitorque2(__global float* __restrict tx, __global float* __restrict ty, __global float* __restrict tz,
                      __global float* __restrict mx, __global float* __restrict my, __global float* __restrict mz,
                      __global float* __restrict Ms_,      float  Ms_mul,
                      __global float* __restrict jz_,      float  jz_mul,
                      __global float* __restrict px_,      float  px_mul,
                      __global float* __restrict py_,      float  py_mul,
                      __global float* __restrict pz_,      float  pz_mul,
                      __global float* __restrict alpha_,   float  alpha_mul,
                      __global float* __restrict pol_,     float  pol_mul,
                      __global float* __restrict lambda_,  float  lambda_mul,
                      __global float* __restrict epsPrime_,float  epsPrime_mul,
                      __global float* __restrict flt_,     float  flt_mul,
                      int N) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float  J = amul(jz_, jz_mul, i);
        float3 p = normalized(vmul(px_, py_, pz_, px_mul, py_mul, pz_mul, i));
        float  Ms           = amul(Ms_, Ms_mul, i);
        float  alpha        = amul(alpha_, alpha_mul, i);
        float  flt          = amul(flt_, flt_mul, i);
        float  pol          = amul(pol_, pol_mul, i);
        float  lambda       = amul(lambda_, lambda_mul, i);
        float  epsilonPrime = amul(epsPrime_, epsPrime_mul, i);

        if (J == 0.0f || Ms == 0.0f) {
            return;
        }

        float beta    = (HBAR / QE) * (J / (flt*Ms) );
        float lambda2 = lambda * lambda;
        float epsilon = pol * lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * dot(p, m));

        float A = beta * epsilon;
        float B = beta * epsilonPrime;

        float gilb     = 1.0f / (1.0f + alpha * alpha);
        float mxpxmFac = gilb * (A + alpha * B);
        float pxmFac   = gilb * (B - alpha * A);

        float3 pxm      = cross(p, m);
        float3 mxpxm    = cross(m, pxm);

        tx[i] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        ty[i] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        tz[i] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
    }
}

