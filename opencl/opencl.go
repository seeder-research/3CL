package opencl

import (
	"bytes"
)

// Type size in bytes
const (
        SIZEOF_FLOAT32    = 4
        SIZEOF_FLOAT64    = 8
        SIZEOF_COMPLEX64  = 8
        SIZEOF_COMPLEX128 = 16
)

var Kernel_codes = make(map[string]*bytes.Buffer)

