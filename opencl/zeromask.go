package opencl

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	eventList := make([]*cl.Event, dst.NComp())
	for c := 0; c < dst.NComp(); c++ {
		eventList[c] = k_zeromask_async(dst.DevPtr(c), unsafe.Pointer(mask), regions.Ptr, N, cfg, [](*cl.Event){dst.GetEvent(c)})
		dst.SetEvent(c, eventList[c])
	}
	err := cl.WaitForEvents(eventList)
	if err != nil {
		fmt.Printf("WaitForEvents in ZeroMask failed: %+v \n", err)
	}
}
