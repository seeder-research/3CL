package kernels

var CLProbDist_Header = `
/* This file is part of clProbDist.
 *
 * Copyright 2015-2016  Pierre L'Ecuyer, Universite de Montreal and Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors:
 *
 *   Nabil Kemerchou <kemerchn@iro.umontreal.ca>    (2015)
 *   David Munger <mungerd@iro.umontreal.ca>        (2015)
 *   Pierre L'Ecuyer <lecuyer@iro.umontreal.ca>     (2015)
 *
 */

/* normal.clh
 *   Device-side API for the normal distribution
 *
 *  In the function declarations of this file, the preprocessor symbol
 *  _CLPROBDIST_NORMAL_OBJ_MEM expands to the selected memory type for
 *  this distribution type.
 */
#ifndef NORMALDIST_CLH
#  define NORMALDIST_CLH

// Begin clProbDist.clh
#  ifndef CLPROBDIST_CLH
#    define CLPROBDIST_CLH

#    ifndef __OPENCL_C_VERSION__
#      error "clProbDist/clProbDist.clh can be included in device code only"
#    endif

typedef float       cl_float;
typedef int         cl_int;
typedef uint        cl_uint;
typedef bool        cl_bool;
#    define CL_TRUE     true
#    define CL_FALSE    false


typedef enum clprobdistStatus_ {
        CLPROBDIST_SUCCESS       =  0,
        CLPROBDIST_INVALID_VALUE = -1,
		CLPROBDIST_OUT_OF_RANGE  = -2
} clprobdistStatus;

#  endif
// End clProbDist.clh

/* Normal distribution object [**device**]
 *
 *  A structure that represents a normal distribution object.
 */
typedef struct _clprobdistNormal clprobdistNormal;

// Begin normal.c.h
#  ifndef CLPROBDIST_PRIVATE_NORMALDIST_CH
#    define CLPROBDIST_PRIVATE_NORMALDIST_CH

//Private variables and Functions
constant cl_float clprobdistNormal_SQRT2 = 1.4142135623730951;

//***********************************************************************
// Implementation of static functions
//***********************************************************************
constant cl_float clprobdistNormal_InvP1[] = {
	0.160304955844066229311E2,
	-0.90784959262960326650E2,
	0.18644914861620987391E3,
	-0.16900142734642382420E3,
	0.6545466284794487048E2,
	-0.864213011587247794E1,
	0.1760587821390590
};

constant cl_float clprobdistNormal_InvQ1[] = {
	0.147806470715138316110E2,
	-0.91374167024260313396E2,
	0.21015790486205317714E3,
	-0.22210254121855132366E3,
	0.10760453916055123830E3,
	-0.206010730328265443E2,
	0.1E1
};

constant cl_float clprobdistNormal_InvP2[] = {
	-0.152389263440726128E-1,
	0.3444556924136125216,
	-0.29344398672542478687E1,
	0.11763505705217827302E2,
	-0.22655292823101104193E2,
	0.19121334396580330163E2,
	-0.5478927619598318769E1,
	0.237516689024448000
};

constant cl_float clprobdistNormal_InvQ2[] = {
	-0.108465169602059954E-1,
	0.2610628885843078511,
	-0.24068318104393757995E1,
	0.10695129973387014469E2,
	-0.23716715521596581025E2,
	0.24640158943917284883E2,
	-0.10014376349783070835E2,
	0.1E1
};

constant cl_float clprobdistNormal_InvP3[] = {
	0.56451977709864482298E-4,
	0.53504147487893013765E-2,
	0.12969550099727352403,
	0.10426158549298266122E1,
	0.28302677901754489974E1,
	0.26255672879448072726E1,
	0.20789742630174917228E1,
	0.72718806231556811306,
	0.66816807711804989575E-1,
	-0.17791004575111759979E-1,
	0.22419563223346345828E-2
};

constant cl_float clprobdistNormal_InvQ3[] = {
	0.56451699862760651514E-4,
	0.53505587067930653953E-2,
	0.12986615416911646934,
	0.10542932232626491195E1,
	0.30379331173522206237E1,
	0.37631168536405028901E1,
	0.38782858277042011263E1,
	0.20372431817412177929E1,
	0.1E1
};

cl_float clprobdistStdNormalInverseCDF(cl_float u, clprobdistStatus* err) {
	/*
	* Returns the inverse of the cdf of the normal distribution.
	* Rational approximations giving 16 decimals of precision.
	* J.M. Blair, C.A. Edwards, J.H. Johnson, "Rational Chebyshev
	* approximations for the Inverse of the Error Function", in
	* Mathematics of Computation, Vol. 30, 136, pp 827, (1976)
	*/

	int i;
	cl_bool negatif;
	cl_float y, z, v, w;
	cl_float x = u;

	if (u < 0.0 || u > 1.0)
	{
		if (err) *err = CLPROBDIST_OUT_OF_RANGE;
		return -1;
	}
	if (u <= 0.0)
		return FLT_MIN;// Double.NEGATIVE_INFINITY;
	if (u >= 1.0)
		return FLT_MAX; // Double.POSITIVE_INFINITY;

	// Transform x as argument of InvErf
	x = 2.0 * x - 1.0;
	if (x < 0.0) {
		x = -x;
		negatif = CL_TRUE;
	}
	else {
		negatif = CL_FALSE;
	}

	if (x <= 0.75) {
		y = x * x - 0.5625;
		v = w = 0.0;
		for (i = 6; i >= 0; i--) {
			v = v * y + clprobdistNormal_InvP1[i];
			w = w * y + clprobdistNormal_InvQ1[i];
		}
		z = (v / w) * x;

	}
	else if (x <= 0.9375) {
		y = x * x - 0.87890625;
		v = w = 0.0;
		for (i = 7; i >= 0; i--) {
			v = v * y + clprobdistNormal_InvP2[i];
			w = w * y + clprobdistNormal_InvQ2[i];
		}
		z = (v / w) * x;

	}
	else {
		if (u > 0.5)
			y = 1.0 / sqrt(-log(1.0 - x));
		else
			y = 1.0 / sqrt(-log(2.0 * u));
		v = 0.0;
		for (i = 10; i >= 0; i--)
			v = v * y + clprobdistNormal_InvP3[i];
		w = 0.0;
		for (i = 8; i >= 0; i--)
			w = w * y + clprobdistNormal_InvQ3[i];
		z = (v / w) / y;
	}

	if (negatif) {
		if (u < 1.0e-105) {
			cl_float RACPI = 1.77245385090551602729;
			w = exp(-z * z) / RACPI;  // pdf
			y = 2.0 * z * z;
			v = 1.0;
			cl_float term = 1.0;
			// Asymptotic series for erfc(z) (apart from exp factor)
			for (i = 0; i < 6; ++i) {
				term *= -(2 * i + 1) / y;
				v += term;
			}
			// Apply 1 iteration of Newton solver to get last few decimals
			z -= u / w - 0.5 * v / z;

		}
		return -(z * clprobdistNormal_SQRT2);

	}
	else
		return z * clprobdistNormal_SQRT2;
}

#  endif

// End normal.c.h

#endif

`
