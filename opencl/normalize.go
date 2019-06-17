package opencl

import (
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Normalize vec to unit length, unless length or vol are zero.
func Normalize(vec, vol *data.Slice) {
	util.Argument(vol == nil || vol.NComp() == 1)
	N := vec.Len()
	cfg := make1DConf(N)
	event := k_normalize2_async(vec.DevPtr(X), vec.DevPtr(Y), vec.DevPtr(Z), vol.DevPtr(0), N, cfg,
		[](*cl.Event){vec.GetEvent(X), vec.GetEvent(Y), vec.GetEvent(Z), vol.GetEvent(X)})

	vec.SetEvent(0, event)
	vec.SetEvent(Y, event)
	vec.SetEvent(Z, event)
	vol.SetEvent(X, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in normalize: %+v \n", err)
	}
}
