package main

import (
	"fmt"
	"math/rand"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"unsafe"
)

func main() {
	var data,origData [128]float32
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
		origData[i] = data[i]
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
	context, queue := opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList
	kernelObj := kernels["square"]
	fmt.Printf("Reached here! \n")
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

	fmt.Printf("Begin first run of buffer tests... \n");

	input, err := context.CreateEmptyBuffer(cl.MemReadWrite, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}
	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(input, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	correct := 0
	for i, v := range data {
		if results[i] == v {
			correct++
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(data))
		return
	}

	fmt.Printf("Modifying single entry of data... \n");

	modTarget := 69;
	modifiedData := make([]float32, 1);
	origData[modTarget] *= -1.0;
	modifiedData[0] = origData[modTarget];
	target := unsafe.Pointer(&modifiedData[0]);
	if _, err :=queue.EnqueueFillBuffer(input, target, 4*len(modifiedData), 4*modTarget, 4*len(modifiedData), nil); err != nil {
		fmt.Printf("EnqueueFillBuffer failed: %+v \n", err)
		return
	}

	fmt.Printf("Reading after first modification \n");

	if _, err := queue.EnqueueReadBufferFloat32(input, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	fmt.Printf("Comparing... \n");

	correct = 0
	for i, v := range origData {
		if results[i] == v {
			correct++
		} else {
			fmt.Printf("data[%d]: %f ; buffer: %f \n", i, v, results[i])
		}
	}

	if correct != len(origData) {
		fmt.Printf("%d/%d correct values \n", correct, len(origData))
		return
	}

	fmt.Printf("Modifying a range of entries of data... \n");

	modTarget = 49;
	modElem := 5;
	modifiedData = make([]float32, modElem);
	for i, _ := range modifiedData {
		idx := modTarget+i;
		origData[idx] *= -0.5;
		modifiedData[i] = origData[idx];
		target = unsafe.Pointer(&modifiedData[i])
		if _, err :=queue.EnqueueFillBuffer(input, target, 4, 4*idx, 4, nil); err != nil {
			fmt.Printf("EnqueueFillBuffer failed: %+v \n", err)
			return
		}
	}

	fmt.Printf("Reading after second modification \n");

	if _, err := queue.EnqueueReadBufferFloat32(input, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	fmt.Printf("Comparing... \n");

	correct = 0
	for i, v := range origData {
		if results[i] == v {
			correct++
		} else {
			fmt.Printf("origData[%d]: %f ; buffer: %f \n", i, v, results[i])
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(origData))
		return
	}

	fmt.Printf("Finished tests on buffer\n")

}

