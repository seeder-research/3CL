package opencl

import (
	"log"
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
)

// Type size in bytes
const (
        SIZEOF_FLOAT32    = 4
        SIZEOF_FLOAT64    = 8
        SIZEOF_COMPLEX64  = 8
        SIZEOF_COMPLEX128 = 16
)

// Assumes kernel arguments set prior to launch
func LaunchKernel(kernname string, gridDim, workDim []int, events []*cl.Event) *cl.Event {
	if KernList[kernname] == nil {
		log.Panic("Kernel "+kernname+" does not exist!")
		return nil
	}
	log.Println("Launching kernel: ", kernname)
	log.Println("gridDim: ", gridDim)
	log.Println("workDim: ", workDim)
	KernEvent, err := ClCmdQueue.EnqueueNDRangeKernel(KernList[kernname], nil, gridDim, workDim, events)
	if err != nil {
		log.Fatal(err)
		return nil
	} else {
		return KernEvent
	}
}

func SetKernelArgWrapper(kernname string, index int, arg interface{}) {
//	log.Printf("Working on index %d \n", index)
	if KernList[kernname] == nil {
		log.Panic("Kernel "+kernname+" does not exist!")
	}
	switch val := arg.(type) {
	default:
		if err := KernList[kernname].SetArg(index, val); err != nil {
			log.Fatal(err)
		}
	case unsafe.Pointer:
		memBufHandle, flag := arg.(unsafe.Pointer)
		if flag {
			if err := KernList[kernname].SetArg(index, (*cl.MemObject)(memBufHandle)); err != nil {
				log.Fatal(err)
			}
		} else {
			log.Fatal("Unable to change argument type to *cl.MemObject")
		}
	case int:
                if err := KernList[kernname].SetArg(index, (int32)(val)); err != nil {
                        log.Fatal(err)
                }
        }
}
