// Kernel to transfer an array of reals to a complex array
__kernel void pack_cmplx(
   __global float2* dst,
   __global float* src,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count) {
      float a0 = src[i];
	  float a1 = 0.0f;
	  float2 b0;
	  b0.x = a0;
	  b0.y = a1;
      dst[i] = b0;
   }
}
