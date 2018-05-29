// Kernel to convert a compact hermitian matrix into a full hermitian matrix
__kernel void hermitian2full(
   __global float2* dst,
   __global float2* src,
   const unsigned int sz,
   const unsigned int count)
{
   int i = get_global_id(0);
   if((i > 0) && (i < count)) {
      dst[sz-i] = src[i];
   }
}
