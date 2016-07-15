#define PREFACTOR ((MUB * MU0) / (2 * QE * GAMMA0))

#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

__kernel void
addzhanglitorque(__global float* __restrict tx, __global float* __restrict ty, __global float* __restrict tz,
                 __global float* __restrict mx, __global float* __restrict my, __global float* __restrict mz,
                 __global float* __restrict jx, __global float* __restrict jy, __global float* __restrict jz,
                 float cx, float cy, float cz,
                 __global float* __restrict bsatLUT, __global  float* __restrict alphaLUT, __global  float* __restrict xiLUT, __global float* __restrict polLUT,
                 __global uint8_t* __restrict regions, int Nx, int Ny, int Nz, uint8_t PBC) {

	int ix = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int iy = get_group_id(1) * get_local_size(1) + get_local_id(1);
	int iz = get_group_id(2) * get_local_size(2) + get_local_id(2);

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	int I = idx(ix, iy, iz);

	uint8_t r = regions[I];
	float alpha = alphaLUT[r];
	float xi    = xiLUT[r];
	float bsat  = bsatLUT[r];
	float pol   = polLUT[r];
	float b = PREFACTOR / (bsat * (1.0f + xi*xi));
	if(bsat == 0.0f) {
		b = 0.0f;
	}
	float Jx = pol*jx[I];
	float Jy = pol*jy[I];
	float Jz = pol*jz[I];

	float3 hspin = make_float3(0.0f, 0.0f, 0.0f); // (u·∇)m
	if (Jx != 0.0f) {
		hspin += (b/cx)*Jx * make_float3(deltax(mx), deltax(my), deltax(mz));
	}
	if (Jy != 0.0f) {
		hspin += (b/cy)*Jy * make_float3(deltay(mx), deltay(my), deltay(mz));
	}
	if (Jz != 0.0f) {
		hspin += (b/cz)*Jz * make_float3(deltaz(mx), deltaz(my), deltaz(mz));
	}

	float3 m      = make_float3(mx[I], my[I], mz[I]);
	float3 torque = (-1.0f/(1.0f + alpha*alpha)) * (
	                    (1.0f+xi*alpha) * cross(m, cross(m, hspin))
	                    +(  xi-alpha) * cross(m, hspin)           );

	// write back, adding to torque
	tx[I] += torque.x;
	ty[I] += torque.y;
	tz[I] += torque.z;
}

