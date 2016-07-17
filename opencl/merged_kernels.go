package opencl

import (
       "github.com/mumax/3cl/opencl/kernels"
)

func GenMergedKernelSource() string {
     return kernels.OpenclProgramSource()
}
