package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/engine"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"math"
	"math/rand"
	"os"
	"unsafe"
)

var (
	Flag_size  = flag.Int("length", 10, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

func main() {
	flag.Parse()
	dataSize := int(*Flag_size)
	if (dataSize < 3) || (dataSize > 16) {
		fmt.Println("argument to -length must be an integer from 4 to 18!")
		os.Exit(-1)
	}
	dataSize = int(math.Pow(float64(3.0),float64(dataSize)))
	// tol := 5e-7

	opencl.Init(*engine.Flag_gpu)

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
	// kernels := opencl.KernList

	/********** Generate and print input array **********/
	fmt.Println("Generating input data...")
	inputs0 := make([]float32, 2*dataSize)
	for j := 0; j < len(inputs0); j++ {
		inputs0[j] = rand.Float32()
	}

	print_iter := 0
	fmt.Println("Input data generated: \n")
	for print_iter < dataSize {
		fmt.Printf("(%f, %f) ", inputs0[2*print_iter], inputs0[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")
	fmt.Println("Moving on... \n")

	/**********  Transfer input to GPU **********/
	fmt.Println("Transferring input data from CPU to GPU...")
	bufX, errC := context.CreateEmptyBuffer(cl.MemReadWrite, dataSize*2*int(unsafe.Sizeof(inputs0[0])))
	if errC != nil {
		fmt.Printf("unable to create input buffer \n")
	}

	bufOut, errCO := context.CreateEmptyBuffer(cl.MemReadWrite, dataSize*2*int(unsafe.Sizeof(inputs0[0])))
	if errCO != nil {
		fmt.Printf("unable to create output buffer \n")
	}

	if _, err := queue.EnqueueWriteBufferFloat32(bufX, true, 0, inputs0[:], nil); err != nil {
		fmt.Printf("failed to write data into buffer \n")
	}

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Input data transfer completed.")

	/********** Set up FFT for power of 2 input **********/
	flag := cl.CLFFTDim1D
	fftPlanHandle, errF := cl.NewCLFFTPlan(context, flag, []int{dataSize})
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

	/********** Bake the FFT plan **********/
	errF = fftPlanHandle.BakePlanSimple([]*cl.CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/********** Execute the baked FFT plan **********/
	_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufX}, []*cl.MemObject{bufOut}, nil)
	if errF != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errF)
	}

	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to flush queue: %+v \n", errF)
	}

	/********** Fetch golden FFT results (power of 2) **********/
	X := make([]float32, len(inputs0))
	_, errF = queue.EnqueueReadBufferFloat32(bufOut, true, 0, X, nil)
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/********** Print golden FFT results to screen **********/
	fmt.Printf("Returned FFT result: \n")
	print_iter = 0
	for print_iter < dataSize {
		fmt.Printf("(%f, %f) ", X[2*print_iter], X[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")
	fmt.Printf("Performing IFFT to confirm... \n")

	/********** Set up inverse FFT to check golden results **********/
	fftPlanHandleI, errFI := cl.NewCLFFTPlan(context, flag, []int{dataSize})
	if errFI != nil {
		fmt.Printf("unable to create new inverse fft plan \n")
	}
	errFI = fftPlanHandleI.SetSinglePrecision()
	if errFI != nil {
		fmt.Printf("unable to set ifft precision \n")
	}
	ArrLayout = cl.NewArrayLayout()
	ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
	errFI = fftPlanHandleI.SetResultOutOfPlace()
	if errFI != nil {
		fmt.Printf("unable to set ifft result location \n")
	}

	/********** Bake inverse FFT plan **********/
	errFI = fftPlanHandleI.BakePlanSimple([]*cl.CommandQueue{queue})
	if errFI != nil {
		fmt.Printf("unable to bake ifft plan: %+v \n", errFI)
	}

	/********** Execute the inverse FFT plan **********/
	_, errFI = fftPlanHandleI.EnqueueBackwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufOut}, []*cl.MemObject{bufX}, nil)
	if errFI != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errFI)
	}

	errFI = queue.Finish()
	if errFI != nil {
		fmt.Printf("unable to flush queue: %+v \n", errFI)
	}

	/********** Fetch result of the inverse FFT plan **********/
	_, errFI = queue.EnqueueReadBufferFloat32(bufX, true, 0, X, nil)
	errFI = queue.Finish()
	if errFI != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/********** Print result of the inverse FFT to screen **********/
	fmt.Printf("Returned IFFT result: \n")
	print_iter = 0
	for print_iter < dataSize {
		fmt.Printf("(%f, %f) ", X[2*print_iter], X[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")
	/********** First run the FFT using clFFT is complete **********/

	fftPlanHandle.Destroy()
	fftPlanHandleI.Destroy()

	/* First Bluestein test...
	   We will try with new length that is double the original length,
	   which is already a power of 2 */

	bluestein_len := 2 * dataSize
	/********** Create the twiddle array **********/
	fmt.Printf("Creating twiddle array... \n")
	twidArray := make([]float32, 2*bluestein_len)
	twidAngle := math.Pi / float64(dataSize)
	twidArray[0] = float32(1.0)
	twidArray[1] = float32(0.0)
	for j := 1; j < dataSize; j++ {
		currAngle := twidAngle * float64(j*j)
		cx, sx := float32(math.Cos(currAngle)), float32(math.Sin(currAngle))
		twidArray[2*j], twidArray[2*(bluestein_len-j)] = cx, cx
		twidArray[2*j+1], twidArray[2*(bluestein_len-j)+1] = sx, sx
	}
	twidArray[2*dataSize], twidArray[2*dataSize+1] = float32(1.0), float32(0.0)

	/********** Print the twiddle array to screen **********/
	fmt.Printf("Twiddle array: \n")
	print_iter = 0
	for print_iter < bluestein_len {
		fmt.Printf("(%f, %f) ", twidArray[2*print_iter], twidArray[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	/********** Copy the twiddle array to GPU **********/
	fmt.Printf("Begin processing twiddle array... \n")

	bufB, errB := context.CreateEmptyBuffer(cl.MemReadWrite, bluestein_len*2*int(unsafe.Sizeof(twidArray[0])))
	if errB != nil {
		fmt.Printf("unable to create buffer for twiddle \n")
	}

	if _, err := queue.EnqueueWriteBufferFloat32(bufB, true, 0, twidArray[:], nil); err != nil {
		fmt.Printf("failed to write twiddle array into buffer \n")
	}

	fmt.Println("Waiting for data transfer to complete...")
	queue.Finish()
	fmt.Println("Twiddle array transfer completed.")

	/********** Create buffer to store FFT result of the twiddle array **********/
	bufBTran, errBTran := context.CreateEmptyBuffer(cl.MemReadWrite, bluestein_len*2*int(unsafe.Sizeof(twidArray[0])))
	if errBTran != nil {
		fmt.Printf("unable to create buffer for twiddle transform \n")
	}

	/********** Perform FFT on the twiddle array **********/
	fftPlanHandle, errF = cl.NewCLFFTPlan(context, flag, []int{bluestein_len})
	if errF != nil {
		fmt.Printf("unable to create new fft plan \n")
	}
	errF = fftPlanHandle.SetSinglePrecision()
	if errF != nil {
		fmt.Printf("unable to set fft precision \n")
	}
	ArrLayout = cl.NewArrayLayout()
	ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
	errF = fftPlanHandle.SetResultOutOfPlace()
	if errF != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	/********** Bake FFT plan for the twiddle array **********/
	errF = fftPlanHandle.BakePlanSimple([]*cl.CommandQueue{queue})
	if errF != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/********** Execute FFT plan for the twiddle array **********/
	fmt.Printf("Executing transform on twiddle array... \n")
	_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufB}, []*cl.MemObject{bufBTran}, nil)
	if errF != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errF)
	}

	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}
	fmt.Printf("Done. \n")

	/********** Extend input array to Bluestein length **********/
	fmt.Printf("Copying input array to Bluestein input... \n")
	bufA, errA := context.CreateEmptyBuffer(cl.MemWriteOnly, bluestein_len*2*int(unsafe.Sizeof(twidArray[0])))
	if errA != nil {
		fmt.Printf("unable to create buffer for Bluestein input \n")
	}

	/********** Copy input array to buffer with Bluestein length **********/
	_, errA = queue.EnqueueCopyBuffer(bufX, bufA, 0, 0, 2*dataSize*int(unsafe.Sizeof(inputs0[0])), nil)
	if errA != nil {
		fmt.Printf("unable to copy input buffer to Bluestein input buffer \n")
	}
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}
	fmt.Printf("Done copying...checking... \n")

	/********** Copy back buffer with Bluestein length **********/
	tmp := make([]float32, 2*bluestein_len)
	_, errF = queue.EnqueueReadBufferFloat32(bufA, true, 0, tmp, nil)
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/********** Print buffer with Bluestein length to screen **********/
	fmt.Printf("Input to Bluestein: \n")
	print_iter = 0
	for print_iter < bluestein_len {
		fmt.Printf("(%f, %f) ", tmp[2*print_iter], tmp[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	/********** Performing pointwise complex multiply of Bluestein buffer with conjugate of twiddle **********/
	fmt.Printf("Converting buffers to slices and perform pointwise complex multiply.... \n")
	inputPtr := make([]unsafe.Pointer, 1)
	inputPtr[0] = unsafe.Pointer(bufA)
	sliceA := data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	inputPtr[0] = unsafe.Pointer(bufB)
	sliceB := data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	opencl.ComplexArrayMul(sliceA, sliceA, sliceB, 1, dataSize, 0)
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}

	/********** Copy back buffer with Bluestein length **********/
	_, errF = queue.EnqueueReadBufferFloat32(bufA, true, 0, tmp, nil)
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to read output buffer: %+v \n", errF)
	}

	/********** Print buffer with Bluestein length to screen **********/
	fmt.Printf("Input to Bluestein: \n")
	print_iter = 0
	for print_iter < bluestein_len {
		fmt.Printf("(%f, %f) ", tmp[2*print_iter], tmp[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")

	/********** Perform FFT on Bluestein buffer **********/
	fmt.Printf("Done multiplying...Performing FFT \n")
	bufATran, errATran := context.CreateEmptyBuffer(cl.MemWriteOnly, bluestein_len*2*int(unsafe.Sizeof(inputs0[0])))
	if errATran != nil {
		fmt.Printf("unable to create buffer for transformed Bluestein input \n")
	}
	_, errF = fftPlanHandle.EnqueueForwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufA}, []*cl.MemObject{bufATran}, nil)
	if errF != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errF)
	}

	/********** Perform complex multiply of both FFT results **********/
	fmt.Printf("Converting buffers to slices and perform pointwise complex multiply.... \n")
	inputPtr[0] = unsafe.Pointer(bufATran)
	sliceA = data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	inputPtr[0] = unsafe.Pointer(bufBTran)
	sliceB = data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	opencl.ComplexArrayMul(sliceA, sliceA, sliceB, 0, bluestein_len, 0)
	fmt.Printf("Done multiplying...Performing inverse FFT \n")

	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}

	/********** Perform inverse FFT of multiplication result **********/
	fftPlanHandleI, errFI = cl.NewCLFFTPlan(context, flag, []int{bluestein_len})
	if errFI != nil {
		fmt.Printf("unable to create new inverse fft plan \n")
	}
	errFI = fftPlanHandleI.SetSinglePrecision()
	if errFI != nil {
		fmt.Printf("unable to set fft precision \n")
	}
	ArrLayout = cl.NewArrayLayout()
	ArrLayout.SetInputLayout(cl.CLFFTLayoutComplexInterleaved)
	ArrLayout.SetOutputLayout(cl.CLFFTLayoutComplexInterleaved)
	errFI = fftPlanHandleI.SetResultOutOfPlace()
	if errFI != nil {
		fmt.Printf("unable to set fft result location \n")
	}

	/********** Bake the inverse FFT plan **********/
	errFI = fftPlanHandleI.BakePlanSimple([]*cl.CommandQueue{queue})
	if errFI != nil {
		fmt.Printf("unable to bake fft plan: %+v \n", errF)
	}

	/********** Execute the inverse FFT plan **********/
	fmt.Printf("Executing transform on twiddle array... \n")
	_, errF = fftPlanHandleI.EnqueueBackwardTransform([]*cl.CommandQueue{queue}, nil, []*cl.MemObject{bufATran}, []*cl.MemObject{bufA}, nil)
	if errF != nil {
		fmt.Printf("unable to enqueue transform: %+v \n", errF)
	}

	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}

	fmt.Printf("Done... \n")

	/********** Perform pointwise multiply of the convolution result with the twiddle array **********/
	fmt.Printf("Converting buffers to slices and perform pointwise complex multiply.... \n")
	inputPtr[0] = unsafe.Pointer(bufA)
	sliceA = data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	inputPtr[0] = unsafe.Pointer(bufB)
	sliceB = data.SliceFromPtrs([3]int{2 * bluestein_len, 1, 1}, data.GPUMemory, inputPtr)
	opencl.ComplexArrayMul(sliceA, sliceA, sliceB, 1, dataSize, 0)
	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}
	fmt.Printf("Done multiplying...Get result \n")

	/********** Obtain Bluestein result from GPU **********/
	outArray := make([]float32, 2*dataSize)
	_, errF = queue.EnqueueReadBufferFloat32(bufA, true, 0, outArray, nil)
	if errF != nil {
		fmt.Printf("unable to read buffer: %+v \n", errF)
	}

	errF = queue.Finish()
	if errF != nil {
		fmt.Printf("unable to finish queue: %+v \n", errF)
	}

	/********** Print Bluestein result to screen **********/
	print_iter = 0
	for print_iter < dataSize {
		fmt.Printf("(%f, %f) ", outArray[2*print_iter], outArray[2*print_iter+1])
		print_iter++
	}
	fmt.Printf("\n")
	fmt.Printf("Finished tests on clFFT\n")
	fftPlanHandle.Destroy()
	fftPlanHandleI.Destroy()
	cl.TeardownCLFFT()

	opencl.ReleaseAndClean()
}
