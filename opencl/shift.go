package opencl

import (
	"fmt"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
func ShiftX(dst, src *data.Slice, shiftX int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	event := k_shiftx_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftX, clampL, clampR, cfg, nil)
        err := cl.WaitForEvents([](*cl.Event){event})
        if err != nil {
                fmt.Printf("WaitForEvents in shiftx failed: %+v \n", err)
        }
}

// Like Shift, but for bytes
func ShiftBytes(dst, src *Bytes, m *data.Mesh, shiftX int, clamp byte) {
	N := m.Size()
	cfg := make3DConf(N)
	event := k_shiftbytes_async(dst.Ptr, src.Ptr, N[X], N[Y], N[Z], shiftX, clamp, cfg, nil)
        err := cl.WaitForEvents([](*cl.Event){event})
        if err != nil {
                fmt.Printf("WaitForEvents in shiftbytes failed: %+v \n", err)
        }
}
