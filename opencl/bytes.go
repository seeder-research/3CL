package opencl

// This file provides GPU byte slices, used to store regions.

import (
	"log"
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// 3D byte slice, used for region lookup.
type Bytes struct {
	Ptr unsafe.Pointer
	Len cl.Size_t
}

// Construct new byte slice with given length,
// initialised to zeros.
func NewBytes(Len cl.Size_t) *Bytes {
	ptr, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, Len)
	if err != nil {
		panic(err)
	}
	zeroPattern := uint8(0)
	var event *cl.Event
	event, err = ClCmdQueue.EnqueueFillBuffer(ptr, unsafe.Pointer(&zeroPattern), 1, 0, cl.Size_t(Len), nil)
	if err != nil {
		panic(err)
	}
	err = cl.WaitForEvents([](*cl.Event){event})
	return &Bytes{unsafe.Pointer(ptr), Len}
}

// Upload src (host) to dst (gpu).
func (dst *Bytes) Upload(src []byte) {
	util.Argument(dst.Len == cl.Size_t(len(src)))
	MemCpyHtoD(dst.Ptr, unsafe.Pointer(&src[0]), cl.Size_t(dst.Len))
}

// Copy on device: dst = src.
func (dst *Bytes) Copy(src *Bytes) {
	util.Argument(dst.Len == src.Len)
	MemCpy(dst.Ptr, src.Ptr, cl.Size_t(dst.Len))
}

// Copy to host: dst = src.
func (src *Bytes) Download(dst []byte) {
	util.Argument(src.Len == cl.Size_t(len(dst)))
	MemCpyDtoH(unsafe.Pointer(&dst[0]), src.Ptr, cl.Size_t(src.Len))
}

// Set one element to value.
// data.Index can be used to find the index for x,y,z.
func (dst *Bytes) Set(index int, value byte) {
	if index < 0 || cl.Size_t(index) >= dst.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	src := value
	event, err := ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst.Ptr), false, cl.Size_t(index), 1, unsafe.Pointer(&src), nil);
	if err != nil {
		panic(err)
	}
	err = cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		log.Panic("WaitForEvents failed in Bytes.Set():", err)
	}
}

// Get one element.
// data.Index can be used to find the index for x,y,z.
func (src *Bytes) Get(index int) byte {
	if index < 0 || cl.Size_t(index) >= src.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	dst := make([]byte, 1)
	event , err := ClCmdQueue.EnqueueReadBufferByte((*cl.MemObject)(src.Ptr), false, cl.Size_t(index), dst, nil);
	if err != nil {
		panic(err)
	}
        err = cl.WaitForEvents([](*cl.Event){event})
        if err != nil {
                log.Panic("WaitForEvents failed in Bytes.Set():", err)
        }
	return dst[0]
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	if b.Ptr != nil {
		tmpObj := (*cl.MemObject)(b.Ptr)
		tmpObj.Release()
	}
	b.Ptr = nil
	b.Len = 0
}
