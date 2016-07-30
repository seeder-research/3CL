package opencl

import (
	"unsafe"
	"fmt"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/timer"
	"github.com/mumax/3cl/util"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, size [3]int) *data.Slice {
	return newSlice(nComp, size, data.GPUMemory)
}

// Make a GPU Slice with nComp components each of size length.
//func NewUnifiedSlice(nComp int, m *data.Mesh) *data.Slice {
//	return newSlice(nComp, m, cu.MemAllocHost, data.UnifiedMemory)
//}

func newSlice(nComp int, size [3]int, memType int8) *data.Slice {
	data.EnableGPU(memFree, memFree, MemCpy, MemCpyDtoH, MemCpyHtoD)
	length := prod(size)
	bytes := int64(length) * SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	initVal := float32(0.0)
	for c := range ptrs {
		tmp_buf, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, cl.Size_t(bytes))
		if err != nil { fmt.Printf("CreateEmptyBuffer failed: %+v \n", err) }
		ptrs[c] = unsafe.Pointer(tmp_buf)
		var fillWait *cl.Event
		fillWait, err = ClCmdQueue.EnqueueFillBuffer(tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, cl.Size_t(bytes), nil)
		if err != nil { fmt.Printf("CreateEmptyBuffer failed: %+v \n", err) }
		err = cl.WaitForEvents([]*cl.Event{fillWait})
		if err != nil { fmt.Printf("Wait for EnqueueFillBuffer failed: %+v \n", err) }
	}
	return data.SliceFromPtrs(size, memType, ptrs)
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { 
	buf := (*cl.MemObject)(ptr)
	buf.Release()
}

func MemCpyDtoH(dst, src unsafe.Pointer, bytes cl.Size_t) []*cl.Event {
	// sync previous kernels
	eventList := make([](*cl.Event),1)
	waitList, err := ClCmdQueue.EnqueueBarrierWithWaitList(nil)
	if err != nil {
		fmt.Printf("EnqueueBarrierWithWaitList failed: %+v \n", err)
		return nil
	}
	eventList[0] = waitList
	err = cl.WaitForEvents(eventList)
	if err != nil {
		fmt.Printf("First WaitForEvents in MemCpyDtoH failed: %+v \n", err)
		return nil
	}
	timer.Start("memcpyDtoH")

	// execute
	eventList[0], err = ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(src), false, 0, bytes, dst, nil)
	if err != nil {
		fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
		return nil
	}

	// sync copy
	err = cl.WaitForEvents(eventList)
	timer.Stop("memcpyDtoH")
	if err != nil {
		fmt.Printf("Second WaitForEvents in MemCpyDtoH failed: %+v \n", err)
		return nil
	}

	return eventList
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes cl.Size_t) []*cl.Event {
	// sync previous kernels
        eventList := make([](*cl.Event),1)
        waitList, err := ClCmdQueue.EnqueueBarrierWithWaitList(nil)
        if err != nil {
                fmt.Printf("EnqueueBarrierWithWaitList failed: %+v \n", err)
                return nil
        }
        eventList[0] = waitList
        err = cl.WaitForEvents(eventList)
        if err != nil {
                fmt.Printf("First WaitForEvents in MemCpyHtoD failed: %+v \n", err)
                return nil
        }
	timer.Start("memcpyHtoD")

	// execute
	eventList[0], err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst), false, 0, bytes, src, nil)
	if err != nil {
		fmt.Printf("EnqueueWriteBuffer failed: %+v \n", err)
		return nil
	}

	// sync copy
        err = cl.WaitForEvents(eventList)
	timer.Stop("memcpyHtoD")
        if err != nil {
                fmt.Printf("Second WaitForEvents in MemCpyHtoD failed: %+v \n", err)
                return nil
        }

	return eventList
}

func MemCpy(dst, src unsafe.Pointer, bytes cl.Size_t) []*cl.Event {
	// sync kernels
        eventList := make([](*cl.Event),1)
        waitList, err := ClCmdQueue.EnqueueBarrierWithWaitList(nil)
        if err != nil {
                fmt.Printf("EnqueueBarrierWithWaitList failed: %+v \n", err)
                return nil
        }
        eventList[0] = waitList
        err = cl.WaitForEvents(eventList)
        if err != nil {
                fmt.Printf("First WaitForEvents in MemCpy failed: %+v \n", err)
                return nil
        }
	timer.Start("memcpy")

	// execute
	eventList[0], err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(src), (*cl.MemObject)(dst), 0, 0, bytes, nil)
	if err != nil {
		fmt.Printf("EnqueueCopyBuffer failed: %+v \n", err)
		return nil
	}

	// sync copy
        err = cl.WaitForEvents(eventList)
	timer.Stop("memcpy")
        if err != nil {
                fmt.Printf("First WaitForEvents in MemCpy failed: %+v \n", err)
                return nil
        }

	returnList := make([]*cl.Event,2)
	returnList[0], returnList[1] = eventList[0], eventList[0]
	return returnList
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
        eventList := make([](*cl.Event),1)
	err := cl.WaitForEvents(nil)
 
	if Synchronous { // debug
		eventList[0], err = ClCmdQueue.EnqueueBarrierWithWaitList(nil)
		err = cl.WaitForEvents(eventList)
                if err != nil {
                        fmt.Printf("First WaitForEvents in MemSet failed: %+v \n", err)
                }
		timer.Start("memset")
	}
	util.Argument(len(val) == s.NComp())
	eventListFill := make([](*cl.Event),len(val))
	for c, v := range val {
		eventListFill[c], err = ClCmdQueue.EnqueueFillBuffer((*cl.MemObject)(s.DevPtr(c)), unsafe.Pointer(&v), SIZEOF_FLOAT32, 0, cl.Size_t(s.Len()*SIZEOF_FLOAT32), nil)
		s.SetEvent(c, eventListFill[c])
		if err != nil {
			fmt.Printf("EnqueueFillBuffer failed: %+v \n", err)
		}
	}
	if Synchronous { //debug
		eventList[0], err = ClCmdQueue.EnqueueBarrierWithWaitList(eventListFill)
		err = cl.WaitForEvents(eventList)
                if err != nil {
                        fmt.Printf("Second WaitForEvents in MemSet failed: %+v \n", err)
                }
		timer.Stop("memset")
	}
}

// Set all elements of all components to zero.
func Zero(s *data.Slice) {
	Memset(s, make([]float32, s.NComp())...)
}

func SetCell(s *data.Slice, comp int, ix, iy, iz int, value float32) {
	SetElem(s, comp, s.Index(ix, iy, iz), value)
}

func SetElem(s *data.Slice, comp int, index int, value float32) {
	f := value
	if _, err := ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(s.DevPtr(comp)), false, cl.Size_t(index*SIZEOF_FLOAT32), SIZEOF_FLOAT32, unsafe.Pointer(&f), nil); err != nil {
		fmt.Printf("EnqueueWriteBuffer failed: %+v \n", err)
		return
	}
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var f float32
        if _, err := ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(s.DevPtr(comp)), false, cl.Size_t(index*SIZEOF_FLOAT32), SIZEOF_FLOAT32, unsafe.Pointer(&f), nil); err != nil {
                fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
        }
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}
