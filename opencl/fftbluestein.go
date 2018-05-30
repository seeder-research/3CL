package opencl

import (
	"fmt"
	"github.com/mumax/3cl/util"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
)

// Convert Hermitian FFT array to full complex array
func Hermitian2Full(dst, src *data.Slice) {
        util.Argument(src.NComp() == 1)
	fmt.Println("In opencl library...running kernel")
        event := k_hermitian2full_async(dst.DevPtr(0), src.DevPtr(0), dst.Len()/2, src.Len()/2, reduceintcfg, [](*cl.Event){src.GetEvent(0), dst.GetEvent(0)})
	dst.SetEvent(0, event)
	src.SetEvent(0, event)
//	fmt.Println("Waiting for event to clear")
//        if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
//                fmt.Printf("WaitForEvents failed in hermitian2full: %+v \n", err)
//        }
}
