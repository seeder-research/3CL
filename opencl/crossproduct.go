package opencl

import (
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

func CrossProduct(dst, a, b *data.Slice) {
	util.Argument(dst.NComp() == 3 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	event := k_crossproduct_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg, [](*cl.Event){dst.GetEvent(0), a.GetEvent(X), a.GetEvent(Y), a.GetEvent(Z),
			b.GetEvent(X), b.GetEvent(Y), b.GetEvent(Z)})

	dst.SetEvent(0, event)
	a.SetEvent(X, event)
	a.SetEvent(Y, event)
	a.SetEvent(Z, event)
	b.SetEvent(X, event)
	b.SetEvent(Y, event)
	b.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in crossproduct: %+v \n", err)
	}
}
