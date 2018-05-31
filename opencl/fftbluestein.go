package opencl

import (
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Convert Hermitian FFT array to full complex array
func Hermitian2Full(dst, src *data.Slice) {
	util.Argument(src.NComp() == dst.NComp())
	var tmpEventList, tmpEventList1 []*cl.Event
	for ii := 0; ii < src.NComp(); ii++ {
		tmpEvent := src.GetEvent(ii)
		if tmpEvent != nil {
			tmpEventList = append(tmpEventList, tmpEvent)
		}
	}
	for ii := 0; ii < src.NComp(); ii++ {
		event := k_hermitian2full_async(dst.DevPtr(ii), src.DevPtr(ii), dst.Len()/2, src.Len()/2, reduceintcfg, tmpEventList)
		dst.SetEvent(ii, event)
		src.SetEvent(ii, event)
		tmpEventList1 = append(tmpEventList1, event)
	}

	if err := cl.WaitForEvents(tmpEventList1); err != nil {
		fmt.Printf("WaitForEvents failed in hermitian2full: %+v \n", err)
	}
}
