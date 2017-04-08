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

    int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
    if (i < N) {

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float  J = (jz_ == NULL) ? (jz_mul) : (jz_mul * jz_[i]);
		float pxx = (px_ == NULL) ? (px_mul) : (px_mul * px_[i]);
		float pyy = (py_ == NULL) ? (py_mul) : (py_mul * py_[i]);
		float pzz = (pz_ == NULL) ? (pz_mul) : (pz_mul * pz_[i]);
        float3 p = normalized(make_float3(pxx, pyy, pzz));
        float  Ms           = (Ms_ == NULL) ? (Ms_mul) : (Ms_mul * Ms_[i]);
        float  alpha        = (alpha_ == NULL) ? (alpha_mul) : (alpha_mul * alpha_[i]);
        float  flt          = (flt_ == NULL) ? (flt_mul) : (flt_mul * flt_[i]);
        float  pol          = (pol_ == NULL) ? (pol_mul) : (pol_mul * pol_[i]);
        float  lambda       = (lambda_ == NULL) ? (lambda_mul) : (lambda_mul * lambda_[i]);
        float  epsilonPrime = (epsPrime_ == NULL) ? (epsPrime_mul) : (epsPrime_mul * epsPrime_[i]);

        if (J == 0.0f || Ms == 0.0f) {
            return;
        }

        float beta    = (HBAR / QE) * (J / (flt*Ms) );
        float lambda2 = lambda * lambda;
        float epsilon = pol * lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * dot(p, m));

        float A = beta * epsilon;
        float B = beta * epsilonPrime;

        float gilb     = 1.0f / (1.0f + alpha * alpha);
        float mxpxmFac = gilb * (A - alpha * B);
        float pxmFac   = gilb * (B - alpha * A);

        float3 pxm      = cross(p, m);
        float3 mxpxm    = cross(m, pxm);

        tx[i] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        ty[i] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        tz[i] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
    }
}

