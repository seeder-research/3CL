package kernels

var SlonczewskiSource = `
__kernel void
addslonczewskitorque(__global float* __restrict tx, __global float* __restrict ty, __global float* __restrict tz,
                     __global float* __restrict mx, __global float* __restrict my, __global float* __restrict mz, __global float* __restrict jz,
                     __global float* __restrict px, __global float* __restrict py, __global float* __restrict pz,
                     __global float* __restrict msatLUT, __global float* __restrict alphaLUT, float flt,
                     __global float* __restrict polLUT, __global float* __restrict lambdaLUT, __global float* __restrict epsilonPrimeLUT,
                     __global uint8_t* __restrict regions, int N) {

	int I =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
	if (I < N) {

		float3 m = make_float3(mx[I], my[I], mz[I]);
		float  J = jz[I];
		float3 p = normalized(make_float3(px[I], py[I], pz[I]));

		// read parameters
		uint8_t region       = regions[I];

		float  Ms           = msatLUT[region];
		float  alpha        = alphaLUT[region];
		float  pol          = polLUT[region];
		float  lambda       = lambdaLUT[region];
		float  epsilonPrime = epsilonPrimeLUT[region];

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

		tx[I] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
		ty[I] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
		tz[I] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
	}
}

`
