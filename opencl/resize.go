package opencl

import (
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"

	"fmt"
)

// Select and resize one layer for interactive output
func Resize(dst, src *data.Slice, layer int) {
	dstsize := dst.Size()
	srcsize := src.Size()
	util.Assert(dstsize[Z] == 1)
	util.Assert(dst.NComp() == 1 && src.NComp() == 1)

	scalex := srcsize[X] / dstsize[X]
	scaley := srcsize[Y] / dstsize[Y]
	util.Assert(scalex > 0 && scaley > 0)

	cfg := make3DConf(dstsize)

	event := k_resize_async(dst.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z],
		src.DevPtr(0), srcsize[X], srcsize[Y], srcsize[Z], layer, scalex, scaley, cfg,
		[](*cl.Event){dst.GetEvent(0), src.GetEvent(0)})
	dst.SetEvent(0, event)
	src.SetEvent(0, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents in Resize failed: %+v \n", err)
	}
}
