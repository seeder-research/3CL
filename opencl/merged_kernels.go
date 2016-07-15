package opencl

import (
       "bytes"
)

func GenMergedKernelSource() string {
     // Build source code entirely
     KernelsSource := new(bytes.Buffer)

     // Write headers
     clhInit()
     for _, keyname := range OCLHeadersList {
          KernelsSource.WriteString(Kernel_headers[keyname])
     }

     // Write actual codes
     for _, keyname := range OCLKernelsList {
          KernelsSource.WriteString(Kernel_codes[keyname])
     }

     return KernelsSource.String()
}
