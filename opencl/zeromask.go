package opencl

import (
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	for c := 0; c < dst.NComp(); c++ {
		event := k_zeromask_async(dst.DevPtr(c), unsafe.Pointer(mask), regions.Ptr, N, cfg, [](*cl.Event){dst.GetEvent(c)})
		dst.SetEvent(c, event)
	}
}
