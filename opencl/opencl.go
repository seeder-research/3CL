package opencl

import (
	"log"
)

// Type size in bytes
const (
        SIZEOF_FLOAT32    = 4
        SIZEOF_FLOAT64    = 8
        SIZEOF_COMPLEX64  = 8
        SIZEOF_COMPLEX128 = 16
)

// Assumes kernel arguments set prior to launch
func LaunchKernel(kernname string, gridDim, workDim []int, events []*cl.Event) *cl.Event {
	if KernList[kernname] == nil {
		log.Panic("Kernel "+kernname+" does not exist!")
	}
	KernEvent, err := ClCmdQueue.EnqueueNDRangeKernel(KernList[kernname], nil, gridDim, workDim, events)
	if err != nil {
		log.Fatal(err)
	} else {
		return KernEvent
	}
}
