package main

import (
	"fmt"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"math/rand"
	"unsafe"
)

func main() {
	var data [1024]float32
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}

	N := 16
	X := make([]float32, 2*N)

	opencl.Init(0)
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

	/* print input array */
	fmt.Printf("\nPerforming fft on an one dimensional array of size N = %d \n", N)
	print_iter := 0
	for print_iter < N {
		x := float32(print_iter)
		y := float32(print_iter * 3)
		X[2*print_iter] = x
		X[2*print_iter+1] = y
		fmt.Printf("(%f, %f) ", x, y)
		print_iter++
	}
	fmt.Printf("\n\nfft result: \n")

	/* Prepare OpenCL memory objects and place data inside them. */
	bufX, errC := context.CreateEmptyBuffer(cl.MemWriteOnly, N*2*int(unsafe.Sizeof(X[0])))
	bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadOnly, N*2*int(unsafe.Sizeof(X[0])))

	if errC != nil {
		fmt.Printf("unable to create input buffer: %+v \n ", errC)
	}
	if errCO != nil {
		fmt.Printf("unable to create output buffer: %+v \n ", errCO)
	}

	if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, X[:], nil); err != nil {
		fmt.Printf("failed to write data into buffer \n")
	}

	flag := cl.CLFFTDim1D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{N})
	if errF != nil {
		fmt.Printf("unable to create new fft plan \n")
	}
	errF = fftPlanHandle.SetSinglePrecision()
	if errF != nil {
		fmt.Printf("unable to set fft precision \n")
	}
	ArrLayout := cl.NewArrayLayout()
	ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
	errF = fftPlanHandle.SetResultOutOfPlace()
	if errF != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	/* Bake the plan. */
	errF = fftPlanHandle.BakePlanSimple([]*cl.CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/* Execute the plan. */
	_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufX}, []*cl.MemObject{bufOut}, nil)
	if errF != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errF)
	}

	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to flush queue: %+v \n", errF)
	}

	/* Fetch results of calculations. */
	_, errF = queue.EnqueueReadBufferFloat32(bufOut, true, 0, X, nil)
	errF = queue.Flush()
	if errF != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/* print output array */
	print_iter = 0
	for print_iter < N {
		fmt.Printf("(%f, %f) ", X[2*print_iter], X[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	cl.TeardownCLFFT()

	fmt.Printf("Begin releasing resources\n")
	for _, krn := range kernels {
		krn.Release()
	}

	bufX.Release()
	bufOut.Release()

	opencl.ReleaseAndClean()
}
