package opencl

import (
       "bytes"
)

func GenMergedKernelSource() string {
     // Build source code entirely
     KernelsSource := new(bytes.Buffer)

     // Write headers
     clhInit()
     for _, codes := range Kernel_headers {
          KernelsSource.WriteString(codes)
     }

     // Write actual codes
     for _, codes := range Kernel_codes {
          KernelsSource.WriteString(codes)
     }

     return KernelsSource.String()
}
