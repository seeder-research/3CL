#define BLOCK_DIM 16

// Tranpose of complex matrix
// Offset is used to specify where the first row, first column entry of the matrix
// is located in linear memory space
__kernel void
cmplx_transpose(__global float2* dataOut, __global float2* dataIn, int offset, int width, int height) {
	
	// width = N (signal length)
	// height = batch_size (number of signals in a batch)

	__local float2 block[BLOCK_DIM * (BLOCK_DIM + 1)];
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);
	if ((xIndex < width) && (yIndex < height)) {
		unsigned int index_in = yIndex * width + xIndex + offset;
		unsigned int Idin = get_local_id(1)*(BLOCK_DIM+1) + get_local_id(0);
		block[Idin] = dataIn[index_in];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Write the transposed matric tile to global memory
	xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);

	if ((xIndex < height) && (yIndex < width)) {
		unsigned int index_out = yIndex * height + xIndex + offset;
		unsigned int Idout = get_local_id(0)*(BLOCK_DIM+1) + get_local_id(1);
		dataOut[index_out] = block[Idout];
	}
}
