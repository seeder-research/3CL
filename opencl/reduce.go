package opencl

import (
	"math"
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
	out, intermed := reduceBuf(0)
	intEvent := k_reducesum_async(in.DevPtr(0), intermed, 0, in.Len(), reduceintcfg, [](*cl.Event){in.GetEvent(0)})
	event := k_reducesum_async(intermed, out, 0, ClCUnits, reducecfg, [](*cl.Event){intEvent})
        if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
                fmt.Printf("WaitForEvents failed in sum: %+v \n", err)
        }
	reduceIntBuffers <- (*cl.MemObject)(intermed)
	return copyback(out)
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	result := float32(0)
	// not async over components
	for c := 0; c < nComp; c++ {
		out, intermed := reduceBuf(0)
		barInt := k_reducedot_async(a.DevPtr(c), b.DevPtr(c), intermed, 0, a.Len(), reduceintcfg, []*cl.Event{a.GetEvent(c), b.GetEvent(c)}) // all components add to intermed
                bar := k_reducesum_async(intermed, out, 0, ClCUnits, reducecfg, []*cl.Event{barInt}) // all components add to out
	        if err := cl.WaitForEvents([]*cl.Event{bar}); err != nil {
                	fmt.Printf("WaitForEvents failed at index %d in dot: %+v \n", c, err)
        	}
		reduceIntBuffers <- (*cl.MemObject)(intermed)
 		result += copyback(out)
	}
	return result
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out, intermed := reduceBuf(0)
	intEvent := k_reducemaxabs_async(in.DevPtr(0), intermed, 0, in.Len(), reduceintcfg, [](*cl.Event){in.GetEvent(0)})
	event := k_reducemaxabs_async(intermed, out, 0, ClCUnits, reducecfg, [](*cl.Event){intEvent})
        if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
                fmt.Printf("WaitForEvents failed in maxabs: %+v \n", err)
        }
	reduceIntBuffers <- (*cl.MemObject)(intermed)
	return copyback(out)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out, intermed := reduceBuf(0)
	intEvent := k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), intermed, 0, v.Len(), reduceintcfg, [](*cl.Event){v.GetEvent(0), v.GetEvent(1), v.GetEvent(2)})
	event := k_reducemaxabs_async(intermed, out, 0, ClCUnits, reducecfg, [](*cl.Event){intEvent})
        if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
                fmt.Printf("WaitForEvents failed in maxvecnorm: %+v \n", err)
        }
	return math.Sqrt(float64(copyback(out)))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out, intermed := reduceBuf(0)
  	intEvent := k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		intermed, 0, x.Len(), reduceintcfg, [](*cl.Event){x.GetEvent(0), x.GetEvent(1), x.GetEvent(2), y.GetEvent(0), y.GetEvent(1), y.GetEvent(2)})
	event := k_reducemaxabs_async(intermed, out, 0, ClCUnits, reducecfg, [](*cl.Event){intEvent})
        if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
                fmt.Printf("WaitForEvents failed in maxvecdiff: %+v \n", err)
        }
	return math.Sqrt(float64(copyback(out)))
}

var reduceBuffers chan (*cl.MemObject) // pool of 1-float OpenCL buffers for reduce
var reduceIntBuffers chan (*cl.MemObject) // pool of 1-float OpenCL buffers for reduce

// return a 1-float and an N-float OPENCL reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) (unsafe.Pointer, unsafe.Pointer) {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	interBuf := <-reduceIntBuffers
	waitEvent, err := ClCmdQueue.EnqueueFillBuffer(buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, cl.Size_t(SIZEOF_FLOAT32), nil)
	if err != nil {
		fmt.Printf("reduceBuf failed: %+v \n", err)
		return nil, nil
	}
	err = cl.WaitForEvents([]*cl.Event{waitEvent})
        if err != nil {
                fmt.Printf("First WaitForEvents in reduceBuf failed: %+v \n", err)
                return nil, nil
        }
	waitEvent, err = ClCmdQueue.EnqueueFillBuffer(interBuf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, cl.Size_t(ClCUnits*SIZEOF_FLOAT32), nil)
        if err != nil {
                fmt.Printf("reduceBuf failed: %+v \n", err)
                return nil, nil
        }
        err = cl.WaitForEvents([]*cl.Event{waitEvent})
        if err != nil {
                fmt.Printf("Seconf WaitForEvents in reduceBuf failed: %+v \n", err)
                return nil, nil
        }
	return (unsafe.Pointer)(buf), (unsafe.Pointer)(interBuf)
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), buf, SIZEOF_FLOAT32)
	reduceBuffers <- (*cl.MemObject)(buf)
	return result
}

// initialize pool of 1-float and N-float OPENCL reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan *cl.MemObject, N)
	reduceIntBuffers = make(chan *cl.MemObject, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(cl.Size_t(1 * SIZEOF_FLOAT32))
		reduceIntBuffers <- MemAlloc(cl.Size_t(ClCUnits * SIZEOF_FLOAT32))
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: []int{REDUCE_BLOCKSIZE, 1, 1}, Block: []int{REDUCE_BLOCKSIZE, 1, 1}}
var reduceintcfg = &config{Grid: []int{8*REDUCE_BLOCKSIZE, 1, 1}, Block: []int{REDUCE_BLOCKSIZE, 1, 1}}
