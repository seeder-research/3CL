package main

import (
	"fmt"
	"math/rand"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

func main() {
	var data [1024]float32
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}

	opencl.Init(0, 0)
	platforms := opencl.ClPlatforms
	fmt.Printf("Discovered platforms: \n")
	for i, p := range platforms {
		fmt.Printf("Platform %d: \n", i)
		fmt.Printf("  Name: %s \n", p.Name())
		fmt.Printf("  Vendor: %s \n", p.Vendor())
		fmt.Printf("  Profile: %s \n", p.Profile())
		fmt.Printf("  Version: %s \n", p.Version())
		fmt.Printf("  Extensions: %s \n", p.Extensions())
	}
	platform := opencl.ClPlatform
	fmt.Printf("In use: \n")
	fmt.Printf("  Vendor: %s \n", platform.Vendor())
	fmt.Printf("  Profile: %s \n", platform.Profile())
	fmt.Printf("  Version: %s \n", platform.Version())
	fmt.Printf("  Extensions: %s \n", platform.Extensions())

	fmt.Printf("Discovered devices: \n")
	devices := opencl.ClDevices
	deviceIndex := -1
	for i, d := range devices {
		if deviceIndex < 0 && d.Type() == cl.DeviceTypeGPU {
			deviceIndex = i
		}
		fmt.Printf("Device %d (%s): %s \n", i, d.Type(), d.Name())
		fmt.Printf("  Address Bits: %d \n", d.AddressBits())
		fmt.Printf("  Available: %+v \n", d.Available())
		fmt.Printf("  Compiler Available: %+v \n", d.CompilerAvailable())
		fmt.Printf("  Double FP Config: %s \n", d.DoubleFPConfig())
		fmt.Printf("  Driver Version: %s \n", d.DriverVersion())
		fmt.Printf("  Error Correction Supported: %+v \n", d.ErrorCorrectionSupport())
		fmt.Printf("  Execution Capabilities: %s \n", d.ExecutionCapabilities())
		fmt.Printf("  Extensions: %s \n", d.Extensions())
		fmt.Printf("  Global Memory Cache Type: %s \n", d.GlobalMemCacheType())
		fmt.Printf("  Global Memory Cacheline Size: %d KB \n", d.GlobalMemCachelineSize()/1024)
		fmt.Printf("  Global Memory Size: %d MB \n", d.GlobalMemSize()/(1024*1024))
		fmt.Printf("  Half FP Config: %s \n", d.HalfFPConfig())
		fmt.Printf("  Host Unified Memory: %+v \n", d.HostUnifiedMemory())
		fmt.Printf("  Image Support: %+v \n", d.ImageSupport())
		fmt.Printf("  Image2D Max Dimensions: %d x %d \n", d.Image2DMaxWidth(), d.Image2DMaxHeight())
		fmt.Printf("  Image3D Max Dimenionns: %d x %d x %d \n", d.Image3DMaxWidth(), d.Image3DMaxHeight(), d.Image3DMaxDepth())
		fmt.Printf("  Little Endian: %+v \n", d.EndianLittle())
		fmt.Printf("  Local Mem Size Size: %d KB \n", d.LocalMemSize()/1024)
		fmt.Printf("  Local Mem Type: %s \n", d.LocalMemType())
		fmt.Printf("  Max Clock Frequency: %d \n", d.MaxClockFrequency())
		fmt.Printf("  Max Compute Units: %d \n", d.MaxComputeUnits())
		fmt.Printf("  Max Constant Args: %d \n", d.MaxConstantArgs())
		fmt.Printf("  Max Constant Buffer Size: %d KB \n", d.MaxConstantBufferSize()/1024)
		fmt.Printf("  Max Mem Alloc Size: %d KB \n", d.MaxMemAllocSize()/1024)
		fmt.Printf("  Max Parameter Size: %d \n", d.MaxParameterSize())
		fmt.Printf("  Max Read-Image Args: %d \n", d.MaxReadImageArgs())
		fmt.Printf("  Max Samplers: %d \n", d.MaxSamplers())
		fmt.Printf("  Max Work Group Size: %d \n", d.MaxWorkGroupSize())
		fmt.Printf("  Max Work Item Dimensions: %d \n", d.MaxWorkItemDimensions())
		fmt.Printf("  Max Work Item Sizes: %d \n", d.MaxWorkItemSizes())
		fmt.Printf("  Max Write-Image Args: %d \n", d.MaxWriteImageArgs())
		fmt.Printf("  Memory Base Address Alignment: %d \n", d.MemBaseAddrAlign())
		fmt.Printf("  Native Vector Width Char: %d \n", d.NativeVectorWidthChar())
		fmt.Printf("  Native Vector Width Short: %d \n", d.NativeVectorWidthShort())
		fmt.Printf("  Native Vector Width Int: %d \n", d.NativeVectorWidthInt())
		fmt.Printf("  Native Vector Width Long: %d \n", d.NativeVectorWidthLong())
		fmt.Printf("  Native Vector Width Float: %d \n", d.NativeVectorWidthFloat())
		fmt.Printf("  Native Vector Width Double: %d \n", d.NativeVectorWidthDouble())
		fmt.Printf("  Native Vector Width Half: %d \n", d.NativeVectorWidthHalf())
		fmt.Printf("  OpenCL C Version: %s \n", d.OpenCLCVersion())
		fmt.Printf("  Profile: %s \n", d.Profile())
		fmt.Printf("  Profiling Timer Resolution: %d \n", d.ProfilingTimerResolution())
		fmt.Printf("  Vendor: %s \n", d.Vendor())
		fmt.Printf("  Version: %s \n", d.Version())
	}
	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["square"]
	totalArgs, err := kernelObj.NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
	} else {
		fmt.Printf("Number of arguments in kernel : %d \n", totalArgs)
	}
	for i := 0; i < totalArgs; i++ {
		name, err := kernelObj.ArgName(i)
		if err == cl.ErrUnsupported {
			break
		} else if err != nil {
			fmt.Printf("GetKernelArgInfo for name failed: %+v \n", err)
			break
		} else {
			fmt.Printf("Kernel arg %d: %s \n", i, name)
		}
	}

	fmt.Printf("Begin first run of square kernel... \n");

	input, err := context.CreateEmptyBuffer(cl.MemReadOnly, cl.Size_t(4*len(data)))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	output, err := context.CreateEmptyBuffer(cl.MemReadOnly, cl.Size_t(4*len(data)))
	if err != nil {
		fmt.Printf("CreateBuffer failed for output: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}
	if err := kernelObj.SetArgs(input, output, uint32(len(data))); err != nil {
		fmt.Printf("SetKernelArgs failed: %+v \n", err)
		return
	}

	local, err := kernelObj.WorkGroupSize(device)
	if err != nil {
		fmt.Printf("WorkGroupSize failed: %+v \n", err)
		return
	}
	fmt.Printf("Work group size: %d \n", local)
	size, _ := kernelObj.PreferredWorkGroupSizeMultiple(nil)
	fmt.Printf("Preferred Work Group Size Multiple: %d \n", size)

	global := len(data)
	d := len(data) % local
	if d != 0 {
		global += local - d
	}
	if _, err := queue.EnqueueNDRangeKernel(kernelObj, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	correct := 0
	for i, v := range data {
		if results[i] == v*v {
			correct++
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(data))
		return
	}

	fmt.Printf("First run of square kernel completed...starting second run \n");

	// Create second set of data to re-run kernel
	var data1 [1024]float32
	for i := 0; i < len(data1); i++ {
		data1[i] = rand.Float32()
	}

	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data1[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}

	if _, err := queue.EnqueueNDRangeKernel(kernelObj, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	results1 := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results1, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	correct = 0
	for i, v := range data1 {
		if results1[i] == v*v {
			correct++
		}
	}

	if correct != len(data1) {
		fmt.Printf("%d/%d correct values \n", correct, len(data1))
		return
	}

	fmt.Printf("Finished tests on square\n")

}

