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
		_, err = ClCmdQueue.EnqueueFillBuffer(tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, cl.Size_t(bytes), nil)
		if err != nil { fmt.Printf("CreateEmptyBuffer failed: %+v \n", err) }
	}
	return data.SliceFromPtrs(size, memType, ptrs)
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { 
	buf := (*cl.MemObject)(ptr)
	buf.Release()
}

func MemCpyDtoH(dst, src unsafe.Pointer, bytes cl.Size_t) {
	ClCmdQueue.Finish() // sync previous kernels
	timer.Start("memcpyDtoH")
	if _, err := ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(src), false, 0, bytes, dst, nil); err != nil {
		fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
		return
	}
	ClCmdQueue.Finish() // sync copy
	timer.Stop("memcpyDtoH")
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes cl.Size_t) {
	ClCmdQueue.Finish() // sync previous kernels
	timer.Start("memcpyHtoD")
	if _, err := ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst), false, 0, bytes, src, nil); err != nil {
		fmt.Printf("EnqueueWriteBuffer failed: %+v \n", err)
		return
	}
	ClCmdQueue.Finish() // sync copy
	timer.Stop("memcpyHtoD")
}

func MemCpy(dst, src unsafe.Pointer, bytes cl.Size_t) {
	ClCmdQueue.Finish()
	timer.Start("memcpy")
	if _, err := ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(src), (*cl.MemObject)(dst), 0, 0, bytes, nil); err != nil {
		fmt.Printf("EnqueueCopyBuffer failed: %+v \n", err)
		return
	}
	ClCmdQueue.Finish()
	timer.Stop("memcpy")
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("memset")
	}
	util.Argument(len(val) == s.NComp())
	for c := range val {
		if _, err := ClCmdQueue.EnqueueFillBuffer((*cl.MemObject)(s.DevPtr(c)), unsafe.Pointer(&val[c]), SIZEOF_FLOAT32, 0, cl.Size_t(s.Len()), nil); err != nil {
			fmt.Printf("EnqueueFillBuffer failed: %+v \n", err)
		}
	}
	if Synchronous { //debug
		ClCmdQueue.Finish()
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
