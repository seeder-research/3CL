package opencl

import (
       "bytes"
)

func GenMergedKernelSource() string {
     // Build source code entirely
     var KernelsSource *bytes.Buffer

     // Write headers
     for _, codes := range Kernel_headers {
          KernelsSource.WriteString(codes)
     }

     // Write actual codes
     for _, codes := range Kernel_codes {
          KernelsSource.WriteString(codes.String())
     }

     return KernelsSource.String()
}
