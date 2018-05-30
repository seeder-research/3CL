package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"math/rand"
)

var (
	Flag_size  = flag.Int("fft", 512, "Specify length of FFT")
	Flag_print = flag.Bool("print", false, "Print out result")
)

func main() {
	flag.Parse()

	opencl.Init(*engine.Flag_gpu)

	platform := opencl.ClPlatform
	fmt.Printf("Platform in use: \n")
	fmt.Printf("  Vendor: %s \n", platform.Vendor())
	fmt.Printf("  Profile: %s \n", platform.Profile())
	fmt.Printf("  Version: %s \n", platform.Version())
	fmt.Printf("  Extensions: %s \n", platform.Extensions())

	fmt.Printf("Device in use: \n")

	d := opencl.ClDevice
	fmt.Printf("Device %d (%s): %s \n", *engine.Flag_gpu, d.Type(), d.Name())
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

	queue := opencl.ClCmdQueue
	//	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["hermitian2full"]
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

	fmt.Printf("Begin first run of hermitian2full kernel... \n")

	// Creating inputs
	fmt.Println("Generating input data...")
	testFFTSize := int(*Flag_size)
	dataSize := testFFTSize / 2
	dataSize += 1
	size := [3]int{2 * dataSize, 1, 1}
	inputs := make([][]float32, 1)
	inputs[0] = make([]float32, size[0])
	for i := 0; i < len(inputs[0]); i++ {
		inputs[0][i] = rand.Float32()
	}

	fmt.Println("Done. Transferring input data from CPU to GPU...")
	cpuArray := data.SliceFromArray(inputs, size)
	gpuBuffer := opencl.Buffer(1, size)
	outBuffer := opencl.Buffer(1, [3]int{2 * testFFTSize, 1, 1})
	outArray := data.NewSlice(1, [3]int{2 * testFFTSize, 1, 1})

	data.Copy(gpuBuffer, cpuArray)

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	fmt.Println("Executing kernel...")
	opencl.Hermitian2Full(outBuffer, gpuBuffer)
	fmt.Println("Waiting for kernel to finish execution...")
	queue.Finish()
	fmt.Println("Execution finished.")

	fmt.Println("Retrieving results...")
	data.Copy(outArray, outBuffer)
	queue.Finish()
	fmt.Println("Done.")
	results := outArray.Host()

	scanFlag := testFFTSize % 2
	cntPt := 1
	if scanFlag > 0 {
		cntPt = dataSize
	} else {
		cntPt = dataSize - 1
	}

	incorrect := 0
	var testVarR0, testVarR1, testVarR2, testVarR3 float32

	// Check the pivots
	testVarR0, testVarR1, testVarR2, testVarR3 = results[0][0], results[0][1], inputs[0][0], inputs[0][1]
	if (testVarR0 == testVarR2) && (testVarR1 == testVarR3) {
	} else {
		fmt.Printf("Error at idx[0]: expect (%f + i*(%f)) but have (%f + i*(%f)) \n", testVarR0, testVarR1, testVarR2, testVarR3)
		incorrect++
	}

	if scanFlag > 0 {
		datIdx := 2 * (dataSize - 1)
		testVarR0, testVarR1, testVarR2, testVarR3 = results[0][datIdx], results[0][datIdx+1], inputs[0][datIdx], inputs[0][datIdx+1]
		if (testVarR0 == testVarR2) && (testVarR1 == testVarR3) {
		} else {
			fmt.Printf("Error at idx[%d]: expect (%f + i*(%f)) but have (%f + i*(%f)) \n", dataSize, testVarR0, testVarR1, testVarR2, testVarR3)
			incorrect++
		}
	}

	// Check the rest of the array
	for ii := 1; ii < cntPt; ii++ {
		reflectedIdx := 2 * (testFFTSize - ii)
		testVarR0, testVarR1, testVarR2, testVarR3 = results[0][2*ii], results[0][2*ii+1], results[0][reflectedIdx], results[0][reflectedIdx+1]
		if (testVarR0 == testVarR2) && (testVarR1 == -1.0*testVarR3) {
		} else {
			fmt.Printf("Error at idx[%d]: expect (%f - i*(%f)) but have (%f + i*(%f)) \n", ii, testVarR0, testVarR1, testVarR2, testVarR3)
			incorrect++
		}
	}

	if *Flag_print {
		for ii := 0; ii < dataSize; ii++ {
			fmt.Printf("result = %f + %f ;\t input: %f + %f\n", results[0][2*ii], results[0][2*ii+1], inputs[0][2*ii], inputs[0][2*ii+1])
		}

		for ii := dataSize; ii < testFFTSize; ii++ {
			fmt.Printf("result = %f + %f ;\n", results[0][2*ii], results[0][2*ii+1])
		}
	}

	if incorrect == 0 {
		fmt.Println("All points correct!")
	} else {
		fmt.Println("Errors were found!")
	}

	fmt.Printf("Finished tests on hermitian2full\n")

	fmt.Printf("freeing resources \n")
	opencl.Recycle(gpuBuffer)
	opencl.Recycle(outBuffer)
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}
