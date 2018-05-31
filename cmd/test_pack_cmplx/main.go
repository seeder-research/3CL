package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"math/rand"
)

var (
	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 1024, "Specify GPU")
	Flag_print = flag.Bool("print", false, "Print out array")
)

func main() {
	flag.Parse()
	data := make([]float32, int(*Flag_size))
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}

	opencl.Init(*Flag_gpu)
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
		fmt.Printf("  Image3D Max Dimensions: %d x %d x %d \n", d.Image3DMaxWidth(), d.Image3DMaxHeight(), d.Image3DMaxDepth())
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
		fmt.Printf("  Preferred Work Group Size: %d \n", opencl.ClPrefWGSz)
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

	kernelObj := kernels["pack_cmplx"]
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

	fmt.Printf("Begin first run of pack_cmplx kernel... \n")

	input, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	output, err := context.CreateEmptyBuffer(cl.MemReadOnly, 8*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for output: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}
	if err := kernelObj.SetArgs(output, input, uint32(len(data))); err != nil {
		fmt.Printf("SetKernelArgs failed: %+v \n", err)
		return
	}

	local := device.MaxWorkGroupSize()
	fmt.Printf("Work group size: %d \n", local)

	global := device.MaxComputeUnits() * local

	if _, err := queue.EnqueueNDRangeKernel(kernelObj, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	results := make([]float32, 2*len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	correct := 0
	for i, v := range data {
		if (results[2*i] == v) && (results[2*i+1] == 0.0) {
			correct++
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(data))
		return
	}

	if *Flag_print {
		for i, v := range data {
			fmt.Printf("Expecting [%d]: (%f + i*(%f)) ; have: (%f + i*(%f)) \n", i, v, float32(0.000), results[2*i], results[2*i+1])
		}
	}

	fmt.Printf("Finished tests on pack_cmplx\n")

	fmt.Printf("freeing resources \n")
	input.Release()
	output.Release()
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}
