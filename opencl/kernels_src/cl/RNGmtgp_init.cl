/* ================================ */
/* mtgp32 sample kernel code        */
/* ================================ */
/**
 * This function sets up initial state by seed.
 * kernel function.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[out] d_status kernel I/O data
 * @param[in] seed initializing seed
 */
__kernel void mtgp32_init_seed_kernel(
    __constant uint* param_tbl,
    __constant uint* temper_tbl,
    __constant uint* single_temper_tbl,
    __constant uint* pos_tbl,
    __constant uint* sh1_tbl,
    __constant uint* sh2_tbl,
    __global uint* d_status,
    uint seed)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local uint status[MTGP32_N];
    mtgp32_t mtgp;
    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    // initialize
    mtgp32_init_state(&mtgp, seed + gid);
    barrier(CLK_LOCAL_MEM_FENCE);

    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid] = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
