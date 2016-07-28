package opencl

import (
	"math"
//	"log"
	"fmt"
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

// Block size for reduce kernels.
const REDUCE_BLOCKSIZE = 512

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	bar, events := make([](*cl.Event), 1), make([](*cl.Event), 1)
	events[0] = in.GetEvent(0)
	bar[0] = k_reducesum_async(in.DevPtr(0), out, 0, in.Len(), reducecfg, events)
	if err := cl.WaitForEvents(bar); err != nil {
		fmt.Printf("WaitForEvents failed in sum: %+v \n", err)
	}
	return copyback(out)
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	out := reduceBuf(0)
	// not async over components
	bar, events := make([](*cl.Event), nComp), make([](*cl.Event), 2)
	for c := 0; c < nComp; c++ {
		events[0], events[1] = a.GetEvent(c), b.GetEvent(c)
		bar[c] = k_reducedot_async(a.DevPtr(c), b.DevPtr(c), out, 0, a.Len(), reducecfg, events) // all components add to out
	}
        if err := cl.WaitForEvents(bar); err != nil {
                fmt.Printf("WaitForEvents failed in dot: %+v \n", err)
        }
	return copyback(out)
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
        bar, events := make([](*cl.Event), 1), make([](*cl.Event), 1)
        events[0] = in.GetEvent(0)
	bar[0] = k_reducemaxabs_async(in.DevPtr(0), out, 0, in.Len(), reducecfg, events)
        if err := cl.WaitForEvents(bar); err != nil {
                fmt.Printf("WaitForEvents failed in maxabs: %+v \n", err)
        }
	return copyback(out)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
        bar, events := make([](*cl.Event), 1), make([](*cl.Event), 3)
        events[0], events[1], events[2] = v.GetEvent(0), v.GetEvent(1), v.GetEvent(2)
	bar[0] = k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducecfg, events)
        if err := cl.WaitForEvents(bar); err != nil {
                fmt.Printf("WaitForEvents failed in maxvecnorm: %+v \n", err)
        }
	return math.Sqrt(float64(copyback(out)))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
        bar, events := make([](*cl.Event), 1), make([](*cl.Event), 6)
        events[0], events[1], events[2] = x.GetEvent(0), x.GetEvent(1), x.GetEvent(2)
        events[3], events[4], events[5] = y.GetEvent(0), y.GetEvent(1), y.GetEvent(2)
  	bar[0] = k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg, events)
        if err := cl.WaitForEvents(bar); err != nil {
                fmt.Printf("WaitForEvents failed in maxvecdiff: %+v \n", err)
        }
	return math.Sqrt(float64(copyback(out)))
}

var reduceBuffers chan (*cl.MemObject) // pool of 1-float OpenCL buffers for reduce

// return a 1-float OPENCL reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	waitEvent, err := ClCmdQueue.EnqueueFillBuffer(buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, cl.Size_t(SIZEOF_FLOAT32), nil)
	if err != nil {
		fmt.Printf("reduceBuf failed: %+v \n", err)
		return nil
	}
	waitEventList := make([](*cl.Event), 1)
	waitEventList[0] = waitEvent
	err = cl.WaitForEvents(waitEventList)
        if err != nil {
                fmt.Printf("WaitForEvents in reduceBuf failed: %+v \n", err)
                return nil
        }
	return (unsafe.Pointer)(buf)
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), buf, SIZEOF_FLOAT32)
	reduceBuffers <- (*cl.MemObject)(buf)
	return result
}

// initialize pool of 1-float OPENCL reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan *cl.MemObject, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(1 * SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: []int{8*REDUCE_BLOCKSIZE, 1, 1}, Block: []int{REDUCE_BLOCKSIZE, 1, 1}}
