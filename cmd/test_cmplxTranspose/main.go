package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

var (
	Flag_size  = flag.Int("fft", 4, "Specify length of FFT")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
)

func MatrixTranspose(dst, src *data.Slice, offset, row, col int) {

	queue := opencl.ClCmdQueue
	fmt.Printf("Begin first run of transpose kernel... \n")

	fmt.Println("Executing kernel...")
	opencl.ComplexMatrixTranspose(dst, src, offset, row, col)
	fmt.Println("Waiting for kernel to finish execution...")
	queue.Finish()
	fmt.Println("Execution finished.")

}

func main() {
	var num_row, num_col int
	num_row = 2
	num_col = 2

	flag.Parse()
	testFFTSize := int(*Flag_size)
	NComponents := int(*Flag_comp)
	if testFFTSize < 4 {
		fmt.Println("argument to -fft must be 4 or greater!")
		os.Exit(-1)
	}
	if (NComponents < 1) || (NComponents > 3) {
		fmt.Println("argument to -components must be 1, 2 or 3!")
		os.Exit(-1)
	}

	//Generating Input data
	fmt.Printf("\n Printing Input array \n")
	ip_size := [3]int{2 * num_row * num_col, 1, 1}
	ipdata := make([][]float32, NComponents)
	for i := 0; i < NComponents; i++ {
		ipdata[i] = make([]float32, ip_size[0])
		for j := 0; j < len(ipdata[i]); j++ {
			ipdata[i][j] = rand.Float32()
			if j%2 == 1 {
				fmt.Printf("(%f, %f)", ipdata[i][j-1], ipdata[i][j])
			}

		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

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
	// fmt.Printf("  Address Bits: %d \n", d.AddressBits())
	// fmt.Printf("  Available: %+v \n", d.Available())
	// fmt.Printf("  Compiler Available: %+v \n", d.CompilerAvailable())
	// fmt.Printf("  Double FP Config: %s \n", d.DoubleFPConfig())
	// fmt.Printf("  Driver Version: %s \n", d.DriverVersion())
	// fmt.Printf("  Error Correction Supported: %+v \n", d.ErrorCorrectionSupport())
	// fmt.Printf("  Execution Capabilities: %s \n", d.ExecutionCapabilities())
	// fmt.Printf("  Extensions: %s \n", d.Extensions())
	// fmt.Printf("  Global Memory Cache Type: %s \n", d.GlobalMemCacheType())
	// fmt.Printf("  Global Memory Cacheline Size: %d KB \n", d.GlobalMemCachelineSize()/1024)
	// fmt.Printf("  Global Memory Size: %d MB \n", d.GlobalMemSize()/(1024*1024))
	// fmt.Printf("  Half FP Config: %s \n", d.HalfFPConfig())
	// fmt.Printf("  Host Unified Memory: %+v \n", d.HostUnifiedMemory())
	// fmt.Printf("  Image Support: %+v \n", d.ImageSupport())
	// fmt.Printf("  Image2D Max Dimensions: %d x %d \n", d.Image2DMaxWidth(), d.Image2DMaxHeight())
	// fmt.Printf("  Image3D Max Dimensions: %d x %d x %d \n", d.Image3DMaxWidth(), d.Image3DMaxHeight(), d.Image3DMaxDepth())
	// fmt.Printf("  Little Endian: %+v \n", d.EndianLittle())
	// fmt.Printf("  Local Mem Size Size: %d KB \n", d.LocalMemSize()/1024)
	// fmt.Printf("  Local Mem Type: %s \n", d.LocalMemType())
	// fmt.Printf("  Max Clock Frequency: %d \n", d.MaxClockFrequency())
	// fmt.Printf("  Max Compute Units: %d \n", d.MaxComputeUnits())
	// fmt.Printf("  Max Constant Args: %d \n", d.MaxConstantArgs())
	// fmt.Printf("  Max Constant Buffer Size: %d KB \n", d.MaxConstantBufferSize()/1024)
	// fmt.Printf("  Max Mem Alloc Size: %d KB \n", d.MaxMemAllocSize()/1024)
	// fmt.Printf("  Max Parameter Size: %d \n", d.MaxParameterSize())
	// fmt.Printf("  Max Read-Image Args: %d \n", d.MaxReadImageArgs())
	// fmt.Printf("  Max Samplers: %d \n", d.MaxSamplers())
	// fmt.Printf("  Max Work Group Size: %d \n", d.MaxWorkGroupSize())
	// fmt.Printf("  Preferred Work Group Size: %d \n", opencl.ClPrefWGSz)
	// fmt.Printf("  Max Work Item Dimensions: %d \n", d.MaxWorkItemDimensions())
	// fmt.Printf("  Max Work Item Sizes: %d \n", d.MaxWorkItemSizes())
	// fmt.Printf("  Max Write-Image Args: %d \n", d.MaxWriteImageArgs())
	// fmt.Printf("  Memory Base Address Alignment: %d \n", d.MemBaseAddrAlign())
	// fmt.Printf("  Native Vector Width Char: %d \n", d.NativeVectorWidthChar())
	// fmt.Printf("  Native Vector Width Short: %d \n", d.NativeVectorWidthShort())
	// fmt.Printf("  Native Vector Width Int: %d \n", d.NativeVectorWidthInt())
	// fmt.Printf("  Native Vector Width Long: %d \n", d.NativeVectorWidthLong())
	// fmt.Printf("  Native Vector Width Float: %d \n", d.NativeVectorWidthFloat())
	// fmt.Printf("  Native Vector Width Double: %d \n", d.NativeVectorWidthDouble())
	// fmt.Printf("  Native Vector Width Half: %d \n", d.NativeVectorWidthHalf())
	// fmt.Printf("  OpenCL C Version: %s \n", d.OpenCLCVersion())
	// fmt.Printf("  Profile: %s \n", d.Profile())
	// fmt.Printf("  Profiling Timer Resolution: %d \n", d.ProfilingTimerResolution())
	// fmt.Printf("  Vendor: %s \n", d.Vendor())
	// fmt.Printf("  Version: %s \n", d.Version())

	queue := opencl.ClCmdQueue
	//	device, context, queue := opencl.ClDevice, opencl.ClCtx, opencl.ClCmdQueue
	kernels := opencl.KernList

	kernelObj := kernels["cmplx_transpose"]
	totalArgs, err := kernelObj.NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: %+v \n", err)
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

	op_size := [3]int{2 * num_col * num_row, 1, 1}
	fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	cpuArray := data.SliceFromArray(ipdata, ip_size)

	fmt.Printf("\n Check 1")
	gpuBuffer := opencl.Buffer(NComponents, ip_size)

	fmt.Printf("\n Check 2")
	outBuffer := opencl.Buffer(NComponents, op_size)

	fmt.Printf("\n Check 3")
	outArray := data.NewSlice(NComponents, op_size)

	fmt.Printf("\n Check 4")

	data.Copy(gpuBuffer, cpuArray)

	fmt.Println("\n Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("\n Input data transfer completed.")

	data.Copy(outArray, gpuBuffer)
	temp_results := outArray.Host()
	for i := 0; i < NComponents; i++ {
		for j := 0; j < op_size[0]; j++ {
			if j%2 == 1 {
				fmt.Printf("(%f, %f)", temp_results[i][j-1], temp_results[i][j])
			}
		}
		fmt.Printf("\n")
	}
	MatrixTranspose(outBuffer, gpuBuffer, 0, num_row, num_col)

	fmt.Println("\n Retrieving results...")
	data.Copy(outArray, outBuffer)
	queue.Finish()
	fmt.Println("\n Done.")
	results := outArray.Host()

	for i := 0; i < NComponents; i++ {
		for j := 0; j < op_size[0]; j++ {
			if j%2 == 1 {
				fmt.Printf("(%f, %f)", results[i][j-1], results[i][j])
			}
		}
		fmt.Printf("\n")
	}

	fmt.Printf("freeing resources \n")
	opencl.Recycle(gpuBuffer)
	opencl.Recycle(outBuffer)
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}
