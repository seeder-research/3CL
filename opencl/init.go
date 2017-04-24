// Package opencl provides GPU interaction
package opencl

import (
	"fmt"
	//	"log"
	"runtime"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

var (
	Version      string                    // OpenCL version
	DevName      string                    // GPU name
	TotalMem     int64                     // total GPU memory
	PlatformInfo string                    // Human-readable OpenCL platform description
	GPUInfo      string                    // Human-readable GPU description
	Synchronous  bool                      // for debug: synchronize stream0 at every kernel launch
	ClPlatforms  []*cl.Platform            // list of platforms available
	ClPlatform   *cl.Platform              // platform the global OpenCL context is attached to
	ClDevices    []*cl.Device              // list of devices global OpenCL context may be associated with
	ClDevice     *cl.Device                // device associated with global OpenCL context
	ClCtx        *cl.Context               // global OpenCL context
	ClCmdQueue   *cl.CommandQueue          // command queue attached to global OpenCL context
	ClProgram    *cl.Program               // handle to program in the global OpenCL context
	KernList     = map[string]*cl.Kernel{} // Store pointers to all compiled kernels
	initialized  = false                   // Initial state defaults to false
	ClCUnits     int                       // Get number of compute units available
	ClWGSize     int                       // Get maximum size of work group per compute unit
	ClPrefWGSz	 int                       // Get preferred work group size of device
)

// Locks to an OS thread and initializes CUDA for that thread.
func Init(gpu, platformId int) {
	if initialized {
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
	PlatformInfo = fmt.Sprint("//   Name: ", PlatformName, "\n//   Vendor: ", PlatformVendor, "\n//   Profile: ", PlatformProfile, "\n//   Version: ", PlatformVersion, "\n")
	ClPlatforms = platforms
	ClPlatform = platform

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
	ClDevices = devices
	ClDevice = device
	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		fmt.Printf("CreateContext failed: %+v \n", err)
	}
	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		fmt.Printf("CreateCommandQueue failed: %+v \n", err)
	}
	program, err := context.CreateProgramWithSource([]string{GenMergedKernelSource()})
	if err != nil {
		fmt.Printf("CreateProgramWithSource failed: %+v \n", err)
	}
	if err := program.BuildProgram(nil, "-cl-std=CL1.2 -cl-kernel-arg-info"); err != nil {
		fmt.Printf("BuildProgram failed: %+v \n", err)
	}

	for _, kernname := range KernelNames {
		KernList[kernname], err = program.CreateKernel(kernname)
		if err != nil {
			fmt.Printf("CreateKernel failed: %+v \n", err)
		}
	}
	ClCtx = context
//	ClCtx.Retain()
	ClCmdQueue = queue
//	ClCmdQueue.Retain()
	ClProgram = program
//	ClProgram.Retain()
	// Set basic configuration for distributing
	// work-items across compute units
	ClCUnits, ClWGSize = 8, REDUCE_BLOCKSIZE
	ClCUnits = ClDevice.MaxComputeUnits()
	ClWGSize = ClDevice.MaxWorkGroupSize()
	reducecfg.Grid[0] = ClWGSize
	reducecfg.Block[0] = ClWGSize
	reduceintcfg.Grid[0] = ClWGSize * ClCUnits
	reduceintcfg.Block[0] = ClWGSize
	ClPrefWGSz, err = KernList["madd2"].PreferredWorkGroupSizeMultiple(ClDevice)
	if err != nil {
		fmt.Printf("PreferredWorkGroupSizeMultiple failed: %+v \n", err)
	}

	data.EnableGPU(memFree, memFree, MemCpy, MemCpyDtoH, MemCpyHtoD)

	fmt.Printf("Initializing clFFT library \n")
	if err := cl.SetupCLFFT(); err != nil {
		fmt.Printf("failed to initialize clFFT \n")
	}
}

func ReleaseAndClean() {
	cl.TeardownCLFFT()
	ClCmdQueue.Release()
	ClProgram.Release()
	ClCtx.Release()
}

// Global stream used for everything
//const stream0 = cu.Stream(0)

// Synchronize the global stream
// This is called before and after all memcopy operations between host and device.
//func Sync() {
//	stream0.Synchronize()
//}
