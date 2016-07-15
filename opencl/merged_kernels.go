package opencl

import (
       "bytes"
)

func GenMergedKernelSource() string {
     KernelsSource = New(*bytes.Buffer)
     for _, codes := range Kernel_codes {
          KernelsSource.WriteString(codes)
     }
     return KernelsSource.String()
}

func grabKernelHeaders() string {
}
