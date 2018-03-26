__kernel void
crossproduct(__global float* __restrict  dstx, __global float* __restrict  dsty, __global float* __restrict  dstz,
           __global float* __restrict ax, __global float* __restrict ay, __global float* __restrict az,
           __global float* __restrict bx, __global float* __restrict by, __global float* __restrict bz,
           int N) {

    int i =  ( get_group_id(1)*get_num_groups(0) + get_group_id(0) ) * get_local_size(0) + get_local_id(0);
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        dstx[i] = cross(A, B).x;
        dsty[i] = cross(A, B).y;
        dstz[i] = cross(A, B).z;
    }
}

