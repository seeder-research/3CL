package opencl

import (
	"log"
//	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func MemAlloc(bytes int) *cl.MemObject {
	memObj, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
	if err == cl.ErrMemObjectAllocationFailure || err == cl.ErrOutOfResources {
		log.Fatal(err)
	}
	if err != nil {
		panic(err)
	}
	return memObj
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}
