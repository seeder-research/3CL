package opencl

import (
	"fmt"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

// multiply: dst[i] = a[i] * b[i]
func Mul(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	eventList := make([]*cl.Event, nComp)
	for c := 0; c < nComp; c++ {
		eventList[c] = k_mul_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg,
			    		   [](*cl.Event){dst.GetEvent(c), a.GetEvent(c), b.GetEvent(c)})
		dst.SetEvent(c, eventList[c])
		a.SetEvent(c, eventList[c])
		b.SetEvent(c, eventList[c])
	}
	err := cl.WaitForEvents(eventList)
	if err != nil { fmt.Printf("WaitForEvents failed in mul: %+v \n", err) }
}

// divide: dst[i] = a[i] / b[i]
// divide-by-zero yields zero.
func Div(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	eventList := make([]*cl.Event, nComp)
	for c := 0; c < nComp; c++ {
		eventList[c] = k_pointwise_div_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg,
						[](*cl.Event){dst.GetEvent(c), a.GetEvent(c), b.GetEvent(c)})
		dst.SetEvent(c, eventList[c])
		a.SetEvent(c, eventList[c])
		b.SetEvent(c, eventList[c])
	}
	err := cl.WaitForEvents(eventList)
	if err != nil { fmt.Printf("WaitForEvents failed in div: %+v \n", err) }
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 *data.Slice, factor1, factor2 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp)
	cfg := make1DConf(N)
	eventList := make([]*cl.Event, nComp)
	for c := 0; c < nComp; c++ {
		eventList[c] = k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, N, cfg,
			[](*cl.Event){dst.GetEvent(c), src1.GetEvent(c), src2.GetEvent(c)})
		dst.SetEvent(c, eventList[c])
		src1.SetEvent(c, eventList[c])
		src2.SetEvent(c, eventList[c])
	}
	err := cl.WaitForEvents(eventList)
	if err != nil {	fmt.Printf("WaitForEvents failed in madd2: %+v \n", err) }
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3 * factor3
func Madd3(dst, src1, src2, src3 *data.Slice, factor1, factor2, factor3 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp)
	cfg := make1DConf(N)
	eventList := make([]*cl.Event, nComp)
	for c := 0; c < nComp; c++ {
		eventList[c] = k_madd3_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, src3.DevPtr(c), factor3, N, cfg,
			[](*cl.Event){dst.GetEvent(c), src1.GetEvent(c),
			src2.GetEvent(c), src3.GetEvent(c)})
		dst.SetEvent(c, eventList[c])
		src1.SetEvent(c, eventList[c])
		src2.SetEvent(c, eventList[c])
		src3.SetEvent(c, eventList[c])
	}
	err := cl.WaitForEvents(eventList)
	if err != nil { fmt.Printf("WaitForEvents failed in madd3: %+v \n", err) }
}
