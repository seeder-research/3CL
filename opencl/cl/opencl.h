/*
  This file is used to point the compiler to the actual opencl.h of the system.
  It is also used to check the version of opencl installed
*/
#include <stdlib.h>

#ifdef __APPLE__
	#include <OpenCL/OpenCL.h>
#else
	#include <CL/opencl.h>
#endif

#ifndef CL_VERSION_1_2
	#error "This package requires OpenCL 1.2"
#endif

