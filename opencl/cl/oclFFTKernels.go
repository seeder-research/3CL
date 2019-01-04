package cl

const oclFFTKernelSrc string = `
// Complex multiplication
// dst[i] = (a[i].x + i*a[i].y)*(b[i].x + i*b[i].y)
// dst[i].x = a[i].x*b[i].x - a[i].y*b[i].y
// dst[i].y = a[i].x*b[i].y + a[i].y*b[i].x
__kernel void
cmplx_mul(__global float2* __restrict  dst, __global float2* __restrict  a, __global float2* __restrict b, const unsigned int conjB, const unsigned int N, const unsigned int offset) {

	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	for (int i = grp_id * grp_sz + local_idx; i < N; i += grp_offset) {
		float2 aa = a[i+offset];
		float2 bb = b[i+offset];
		float xx, yy, xy, yx;
		float2 dd;
		xx = aa.x * bb.x;
		yy = aa.y * bb.y;
		xy = aa.x * bb.y;
		yx = aa.y * bb.x;
		if (conjB == 0) {
			dd.x = xx - yy;
			dd.y = xy + yx;
		} else {
			dd.x = xx + yy;
			dd.y = yx - xy;
		}
	dst[i+offset] = dd;
	}
}

// Kernel to get an array of reals from a complex array
__kernel void
compress_cmplx(__global float* dst, __global float2* src, const unsigned int count, const unsigned int iOffset, const unsigned int oOffset)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	while(global_idx < count) {
		float2 a0 = src[global_idx + iOffset];
		dst[global_idx + oOffset] = a0.x;
		global_idx += grp_offset;
	}
}

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix transpose with OpenCL
* Device code.
*/

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

// Tranpose of complex matrix
// Offset is used to specify where the first row, first column entry of the matrix
// is located in linear memory space
__kernel void
finaltwiddlefact(__global float2* dataOut, int origlength, int extenlength, int fftdirec, int offset) {
	int mul_sign = -1;
	// width = N (signal length)
	// height = batch_size (number of signals in a batch)
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	if (fftdirec != 1) {
		mul_sign = 1;
	} else {
		mul_sign = -1;
	}
	for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += 1) {
	float2 dd;
	float theta;
	theta = i * i * M_PI / origlength;
	dd.x = cos(theta);
	dd.y = mul_sign * sin(theta);
	dataOut[i+offset] = dd;
	}
}

// Kernel to convert a compact hermitian matrix into a full hermitian matrix
__kernel void
hermitian2full(__global float2* __restrict dst, __global float2* __restrict src, const unsigned int sz, const unsigned int count, __local float* scratchR, __local float* scratchI)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int currGrpOffset = grp_id * grp_sz;
	int global_idx = currGrpOffset + local_idx; // Calculate global index of work-item
	int ngrp = get_num_groups(0); // Get number of groups
	int grp_offset = ngrp * grp_sz; // Offset for memory access

	int currCnt = count - currGrpOffset; // Track how many items we have left to update in the array
	int currSize = (currCnt < grp_sz) ? currCnt : grp_sz; // Default value of group size.
	float2 tmpR0;

	while (currCnt > 0) {
		if (global_idx < count) {
			tmpR0 = src[global_idx]; // Grab src array element
			dst[global_idx] = tmpR0; // Copy the first half from src to dst
			tmpR0.y *= -1.0f;

			// Place reversed in local memory
			int tmp_idx = currSize - local_idx - 1;
			if (tmp_idx >= 0) {
				scratchR[tmp_idx] = tmpR0.x;
				scratchI[tmp_idx] = tmpR0.y;
			}
		}

		// Synchronize work-items
		barrier(CLK_LOCAL_MEM_FENCE);

		if (global_idx < count) {
			// Retrieve in order from local memory and store in global memory
			tmpR0.x = scratchR[local_idx];
			tmpR0.y = scratchI[local_idx];
			int outIdx = sz - currGrpOffset - currSize + local_idx + 1;
			if ((outIdx < sz) && (outIdx >= count)) {
				dst[outIdx] = tmpR0;
			}
		}

		// Synchronize work-items
		barrier(CLK_LOCAL_MEM_FENCE);

		grp_id += ngrp;
		global_idx += grp_offset;

		currGrpOffset = grp_id * grp_sz;
		currCnt = count - currGrpOffset; // Track how many items we have left to update in the array
		currSize = (currCnt < grp_sz) ? currCnt : grp_sz;
	}
}

__kernel void
multwiddlefact(__global float2* dataOut, int origlength, int extenlength, int fftdirec, int offset) {

    int mul_sign;
    // width = N (signal length)
	// height = batch_size (number of signals in a batch)
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    if (fftdirec == 1) {
        mul_sign = 1;
    } else {
        mul_sign = -1;
    }
    //for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += grp_offset) {
    for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += 1) {
        float2 dd;
	    float theta;
        float tempsin;

        if (i < origlength) {
            theta = i * i * M_PI / origlength;
            dd.x = cos(theta);
            tempsin = sin(theta);
            dd.y = mul_sign * tempsin;
        } else if (i >(extenlength - origlength) && i < extenlength) {
            theta = (i - extenlength) * (i - extenlength) * ONEPI / origlength;
            dd.x = cos(theta);
            tempsin = sin(theta);
            dd.y = mul_sign * tempsin;
        } else {
            dd.x = 0;
            dd.y = 0;
        }

	    
	    dataOut[i+offset] = dd;
    }
}

// Kernel to transfer an array of reals to a complex array
__kernel void
pack_cmplx(__global float2* dst, __global float* src, const unsigned int count, const unsigned int iOffset, const unsigned int oOffset)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	while(global_idx < count) {
		float a0 = src[global_idx + iOffset];
		float2 b0;
		b0.x = a0;
		b0.y = 0.0f;
		dst[global_idx + oOffset] = b0;
		global_idx += grp_offset;
	}
}

`

var KernelNames = []string{"cmplx_mul", "compress_cmplx", "cmplx_transpose", "finaltwiddlefact", "hermitian2full", "multwiddlefact", "pack_cmplx"}

func createOclFFTProgram(ctx *Context) (*Program, error) {
	return ctx.CreateProgramWithSource([]string{oclFFTKernelSrc})
}
