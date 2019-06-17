package opencl

import (
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// dst = sqrt(dot(a, a)),
func VecNorm(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	event := k_vecnorm_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		N, cfg, [](*cl.Event){dst.GetEvent(0), a.GetEvent(X), a.GetEvent(Y), a.GetEvent(Z)})
	dst.SetEvent(0, event)
	a.SetEvent(X, event)
	a.SetEvent(Y, event)
	a.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents in VecNorm failed: %+v \n", err)
	}
}
