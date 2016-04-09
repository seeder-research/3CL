// Package opencl provides GPU interaction
package opencl

import (
	"fmt"
//	"log"
	"runtime"

	"go-opencl/opencl/cl"
	"go-opencl/opencl/kernels"
//	"go-opencl/util"
)

var (
	Version     	string			// OpenCL version
	DevName     	string     		// GPU name
	TotalMem    	int64      		// total GPU memory
	PlatformInfo	string     		// Human-readable OpenCL platform description
	GPUInfo     	string     		// Human-readable GPU description
	Synchronous 	bool       		// for debug: synchronize stream0 at every kernel launch
	ClCtx       	cl.Context 		// global OpenCL context
	ClProgram   	cl.Program		// handle to program in the global OpenCL context
	KernList    	map[string]*cl.Kernel	// Store pointers to all compiled kernels
)

// Locks to an OS thread and initializes CUDA for that thread.
func Init(gpu, platformId int) {
	if ClCtx.ContextQuery() != nil {
		fmt.Printf("Already initialized \n")
		return // needed for tests
	}

	runtime.LockOSThread()
	platforms, err := cl.GetPlatforms()
	if err != nil {
		fmt.Printf("Failed to get platforms: %+v \n", err)
	}

	fmt.Printf("// Platform %d: \n", platformId)
	platform := platforms[platformId]

	PlatformName := platform.Name()
	PlatformVendor := platform.Vendor()
	PlatformProfile := platform.Profile()
	PlatformVersion := platform.Version()
	PlatformInfo = fmt.Sprint("//   Name: ", PlatformName, "\n//   Vendor: ", PlatformVendor, "\n//   Profile: ", PlatformProfile, "\n//   Version: ", PlatformVersion,"\n")

	devices, err := platform.GetDevices(cl.DeviceTypeGPU)
	if err != nil {
		fmt.Printf("Failed to get devices: %+v \n", err)
		return
	}
	if len(devices) == 0 {
		fmt.Printf("GetDevices returned no devices \n")
		return
	}
	deviceIndex := -1

	if gpu < len(devices) {
	        deviceIndex = gpu
	} else {
	        fmt.Println("GPU choice not selectable... falling back to first GPU found!")
		deviceIndex = 0
	}

	if deviceIndex < 0 {
	   	deviceIndex = 0
	}

	DevName = devices[deviceIndex].Name()
	TotalMem = devices[deviceIndex].GlobalMemSize()
	Version = devices[deviceIndex].OpenCLCVersion()
	GPUInfo = fmt.Sprint("OpenCL C Version ", Version, "\n// GPU: ", DevName, "(", (TotalMem)/(1024*1024), "MB) \n")
	device := devices[deviceIndex]
	ClCtx, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		fmt.Printf("CreateContext failed: %+v \n", err)
	}
	ClProgram, err := ClCtx.CreateProgramWithSource([]string{kernels.GenMergedKernelSource()})
	if err != nil {
		fmt.Printf("CreateProgramWithSource failed: %+v \n", err)
	}
	if err := ClProgram.BuildProgram(nil, "-cl-std=CL1.2 -cl-kernel-arg-info"); err != nil {
		fmt.Printf("BuildProgram failed: %+v \n", err)
	}

	KernList = map[string]*cl.Kernel{}
	for i0 := range kernels.KernelsList {
		kernName := kernels.KernelsList[i0]
		KernList[kernName], err = ClProgram.CreateKernel(kernName)
		if err != nil {
		       fmt.Printf("CreateKernel failed: %+v \n", err)
		}
	}

}

// Global stream used for everything
//const stream0 = cu.Stream(0)

// Synchronize the global stream
// This is called before and after all memcopy operations between host and device.
//func Sync() {
//	stream0.Synchronize()
//}
