package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
)

var (
	Flag_platform		= flag.Int("platform", 0, "Specify OpenCL platform")
	Flag_gpu			= flag.Int("gpu", 0, "Specify GPU")
	Flag_count			= flag.Int("count", 1, "Number of data points")
)

// Calculate the FFT using slow function
func golden_fft7(dataIn []float32, N int) []float32 {
	arr := dataIn
	out_arr := make([]float32, 14*N)
	for i := 0; i < N; i++ {
		idx := 2*i
		in0_r, in0_i := arr[idx], arr[idx+1]
		in1_r, in1_i := arr[idx+2*N], arr[idx+1+2*N]
		in2_r, in2_i := arr[idx+4*N], arr[idx+1+4*N]
		in3_r, in3_i := arr[idx+6*N], arr[idx+1+6*N]
		in4_r, in4_i := arr[idx+8*N], arr[idx+1+8*N]
		in5_r, in5_i := arr[idx+10*N], arr[idx+1+10*N]
		in6_r, in6_i := arr[idx+12*N], arr[idx+1+12*N]
		cs1, sn1 := float32(math.Cos(math.Pi*float64(2.0/7.0))),  float32(math.Sin(math.Pi*float64(2.0/7.0)))
		cs2, sn2 := float32(math.Cos(math.Pi*float64(4.0/7.0))),  float32(math.Sin(math.Pi*float64(4.0/7.0)))
		cs3, sn3 := float32(math.Cos(math.Pi*float64(6.0/7.0))),  float32(math.Sin(math.Pi*float64(6.0/7.0)))

		out_arr[idx] = in0_r + in1_r + in2_r + in3_r + in4_r + in5_r + in6_r
		out_arr[idx+1] = in0_i + in1_i + in2_i + in3_i + in4_i + in5_i + in6_i
		out_arr[idx+2*N]   = in0_r + (cs1*in1_r + sn1*in1_i) + (cs2*in2_r + sn2*in2_i) + (cs3*in3_r + sn3*in3_i) + (cs3*in4_r - sn3*in4_i) + (cs2*in5_r - sn2*in5_i) + (cs1*in6_r - sn1*in6_i)
		out_arr[idx+2*N+1] = in0_i + (cs1*in1_i - sn1*in1_r) + (cs2*in2_i - sn2*in2_r) + (cs3*in3_i - sn3*in3_r) + (cs3*in4_i + sn3*in4_r) + (cs2*in5_i + sn2*in5_r) + (cs1*in6_i + sn1*in6_r)
		out_arr[idx+4*N]   = in0_r + (cs2*in1_r + sn2*in1_i) + (cs3*in2_r - sn3*in2_i) + (cs1*in3_r - sn1*in3_i) + (cs1*in4_r + sn1*in4_i) + (cs3*in5_r + sn3*in5_i) + (cs2*in6_r - sn2*in6_i)
		out_arr[idx+4*N+1] = in0_i + (cs2*in1_i - sn2*in1_r) + (cs3*in2_i + sn3*in2_r) + (cs1*in3_i + sn1*in3_r) + (cs1*in4_i - sn1*in4_r) + (cs3*in5_i - sn3*in5_r) + (cs2*in6_i + sn2*in6_r)
		out_arr[idx+6*N]   = in0_r + (cs3*in1_r + sn3*in1_i) + (cs1*in2_r - sn1*in2_i) + (cs2*in3_r + sn2*in3_i) + (cs2*in4_r - sn2*in4_i) + (cs1*in5_r + sn1*in5_i) + (cs3*in6_r - sn3*in6_i)
		out_arr[idx+6*N+1] = in0_i + (cs3*in1_i - sn3*in1_r) + (cs1*in2_i + sn1*in2_r) + (cs2*in3_i - sn2*in3_r) + (cs2*in4_i + sn2*in4_r) + (cs1*in5_i - sn1*in5_r) + (cs3*in6_i + sn3*in6_r)
		out_arr[idx+8*N]   = in0_r + (cs3*in1_r - sn3*in1_i) + (cs1*in2_r + sn1*in2_i) + (cs2*in3_r - sn2*in3_i) + (cs2*in4_r + sn2*in4_i) + (cs1*in5_r - sn1*in5_i) + (cs3*in6_r + sn3*in6_i)
		out_arr[idx+8*N+1] = in0_i + (cs3*in1_i + sn3*in1_r) + (cs1*in2_i - sn1*in2_r) + (cs2*in3_i + sn2*in3_r) + (cs2*in4_i - sn2*in4_r) + (cs1*in5_i + sn1*in5_r) + (cs3*in6_i - sn3*in6_r)
		out_arr[idx+10*N]   = in0_r + (cs2*in1_r - sn2*in1_i) + (cs3*in2_r + sn3*in2_i) + (cs1*in3_r + sn1*in3_i) + (cs1*in4_r - sn1*in4_i) + (cs3*in5_r - sn3*in5_i) + (cs2*in6_r + sn2*in6_i)
		out_arr[idx+10*N+1] = in0_i + (cs2*in1_i + sn2*in1_r) + (cs3*in2_i - sn3*in2_r) + (cs1*in3_i - sn1*in3_r) + (cs1*in4_i + sn1*in4_r) + (cs3*in5_i + sn3*in5_r) + (cs2*in6_i - sn2*in6_r)
		out_arr[idx+12*N]   = in0_r + (cs1*in1_r - sn1*in1_i) + (cs2*in2_r - sn2*in2_i) + (cs3*in3_r - sn3*in3_i) + (cs3*in4_r + sn3*in4_i) + (cs2*in5_r + sn2*in5_i) + (cs1*in6_r + sn1*in6_i)
		out_arr[idx+12*N+1] = in0_i + (cs1*in1_i + sn1*in1_r) + (cs2*in2_i + sn2*in2_r) + (cs3*in3_i + sn3*in3_r) + (cs3*in4_i - sn3*in4_r) + (cs2*in5_i - sn2*in5_r) + (cs1*in6_i - sn1*in6_r)
	}
	return out_arr
}

func golden_ifft7(dataIn []float32, N int) []float32 {
	arr := dataIn
	out_arr := make([]float32, 14*N)
	for i := 0; i < N; i++ {
		idx := 2*i
		in0_r, in0_i := arr[idx], arr[idx+1]
		in1_r, in1_i := arr[idx+2*N], arr[idx+1+2*N]
		in2_r, in2_i := arr[idx+4*N], arr[idx+1+4*N]
		in3_r, in3_i := arr[idx+6*N], arr[idx+1+6*N]
		in4_r, in4_i := arr[idx+8*N], arr[idx+1+8*N]
		in5_r, in5_i := arr[idx+10*N], arr[idx+1+10*N]
		in6_r, in6_i := arr[idx+12*N], arr[idx+1+12*N]
		cs1, sn1 := float32(math.Cos(math.Pi*float64(-2.0/7.0))),  float32(math.Sin(math.Pi*float64(-2.0/7.0)))
		cs2, sn2 := float32(math.Cos(math.Pi*float64(-4.0/7.0))),  float32(math.Sin(math.Pi*float64(-4.0/7.0)))
		cs3, sn3 := float32(math.Cos(math.Pi*float64(-6.0/7.0))),  float32(math.Sin(math.Pi*float64(-6.0/7.0)))

		out_arr[idx] = (in0_r + in1_r + in2_r + in3_r + in4_r + in5_r + in6_r) / 7.0
		out_arr[idx+1] = (in0_i + in1_i + in2_i + in3_i + in4_i + in5_i + in6_i) / 7.0
		out_arr[idx+2*N]   = (in0_r + (cs1*in1_r + sn1*in1_i) + (cs2*in2_r + sn2*in2_i) + (cs3*in3_r + sn3*in3_i) + (cs3*in4_r - sn3*in4_i) + (cs2*in5_r - sn2*in5_i) + (cs1*in6_r - sn1*in6_i)) / 7.0
		out_arr[idx+2*N+1] = (in0_i + (cs1*in1_i - sn1*in1_r) + (cs2*in2_i - sn2*in2_r) + (cs3*in3_i - sn3*in3_r) + (cs3*in4_i + sn3*in4_r) + (cs2*in5_i + sn2*in5_r) + (cs1*in6_i + sn1*in6_r)) / 7.0
		out_arr[idx+4*N]   = (in0_r + (cs2*in1_r + sn2*in1_i) + (cs3*in2_r - sn3*in2_i) + (cs1*in3_r - sn1*in3_i) + (cs1*in4_r + sn1*in4_i) + (cs3*in5_r + sn3*in5_i) + (cs2*in6_r - sn2*in6_i)) / 7.0
		out_arr[idx+4*N+1] = (in0_i + (cs2*in1_i - sn2*in1_r) + (cs3*in2_i + sn3*in2_r) + (cs1*in3_i + sn1*in3_r) + (cs1*in4_i - sn1*in4_r) + (cs3*in5_i - sn3*in5_r) + (cs2*in6_i + sn2*in6_r)) / 7.0
		out_arr[idx+6*N]   = (in0_r + (cs3*in1_r + sn3*in1_i) + (cs1*in2_r - sn1*in2_i) + (cs2*in3_r + sn2*in3_i) + (cs2*in4_r - sn2*in4_i) + (cs1*in5_r + sn1*in5_i) + (cs3*in6_r - sn3*in6_i)) / 7.0
		out_arr[idx+6*N+1] = (in0_i + (cs3*in1_i - sn3*in1_r) + (cs1*in2_i + sn1*in2_r) + (cs2*in3_i - sn2*in3_r) + (cs2*in4_i + sn2*in4_r) + (cs1*in5_i - sn1*in5_r) + (cs3*in6_i + sn3*in6_r)) / 7.0
		out_arr[idx+8*N]   = (in0_r + (cs3*in1_r - sn3*in1_i) + (cs1*in2_r + sn1*in2_i) + (cs2*in3_r - sn2*in3_i) + (cs2*in4_r + sn2*in4_i) + (cs1*in5_r - sn1*in5_i) + (cs3*in6_r + sn3*in6_i)) / 7.0
		out_arr[idx+8*N+1] = (in0_i + (cs3*in1_i + sn3*in1_r) + (cs1*in2_i - sn1*in2_r) + (cs2*in3_i + sn2*in3_r) + (cs2*in4_i - sn2*in4_r) + (cs1*in5_i + sn1*in5_r) + (cs3*in6_i - sn3*in6_r)) / 7.0
		out_arr[idx+10*N]   = (in0_r + (cs2*in1_r - sn2*in1_i) + (cs3*in2_r + sn3*in2_i) + (cs1*in3_r + sn1*in3_i) + (cs1*in4_r - sn1*in4_i) + (cs3*in5_r - sn3*in5_i) + (cs2*in6_r + sn2*in6_i)) / 7.0
		out_arr[idx+10*N+1] = (in0_i + (cs2*in1_i + sn2*in1_r) + (cs3*in2_i - sn3*in2_r) + (cs1*in3_i - sn1*in3_r) + (cs1*in4_i + sn1*in4_r) + (cs3*in5_i + sn3*in5_r) + (cs2*in6_i - sn2*in6_r)) / 7.0
		out_arr[idx+12*N]   = (in0_r + (cs1*in1_r - sn1*in1_i) + (cs2*in2_r - sn2*in2_i) + (cs3*in3_r - sn3*in3_i) + (cs3*in4_r + sn3*in4_i) + (cs2*in5_r + sn2*in5_i) + (cs1*in6_r + sn1*in6_i)) / 7.0
		out_arr[idx+12*N+1] = (in0_i + (cs1*in1_i + sn1*in1_r) + (cs2*in2_i + sn2*in2_r) + (cs3*in3_i + sn3*in3_r) + (cs3*in4_i - sn3*in4_r) + (cs2*in5_i - sn2*in5_r) + (cs1*in6_i - sn1*in6_r)) / 7.0
	}
	return out_arr
}

func main() {
	flag.Parse()

	nElem := *Flag_count
	testSz := nElem * (int)(14)
	data := make([]float32, testSz)
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}
	gold_res := golden_fft7(data, nElem)
	gold_ret := golden_ifft7(gold_res, nElem)
	
	opencl.Init(*Flag_platform, *Flag_gpu)
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

	kernelObj := kernels["fft7_c2c_long_interleaved_oop"]
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

	fmt.Printf("Begin first run of fft7_c2c_long_interleaved_oop kernel... \n");

	input, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	output, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for output: %+v \n", err)
		return
	}
	rev_in, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for input: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}
	if err := kernelObj.SetArgs(input, output, uint32(nElem), uint32(1), uint32(5)); err != nil {
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

	fmt.Printf("First run of fft7_c2c_long_interleaved_oop kernel completed and starting inverse to check... \n");

	kernelObj1 := kernels["ifft7_c2c_long_interleaved_oop"]
	totalArgs, err = kernelObj1.NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
	} else {
		fmt.Printf("Number of arguments in kernel : %d \n", totalArgs)
	}
	for i := 0; i < totalArgs; i++ {
		name, err := kernelObj1.ArgName(i)
		if err == cl.ErrUnsupported {
			break
		} else if err != nil {
			fmt.Printf("GetKernelArgInfo for name failed: %+v \n", err)
			break
		} else {
			fmt.Printf("Kernel arg %d: %s \n", i, name)
		}
	}

	if err = kernelObj1.SetArgs(output, rev_in, uint32(nElem), uint32(1), uint32(5)); err != nil {
		fmt.Printf("SetKernelArgs failed: %+v \n", err)
		return
	}

	local, err = kernelObj1.WorkGroupSize(device)
	if err != nil {
		fmt.Printf("WorkGroupSize failed: %+v \n", err)
		return
	}
	fmt.Printf("Work group size: %d \n", local)
	size, _ = kernelObj1.PreferredWorkGroupSizeMultiple(nil)
	fmt.Printf("Preferred Work Group Size Multiple: %d \n", size)

	fmt.Printf("Begin first run of ifft7_c2c_long_interleaved_oop kernel... \n");

	if _, err := queue.EnqueueNDRangeKernel(kernelObj1, nil, []int{global}, []int{local}, nil); err != nil {
		fmt.Printf("EnqueueNDRangeKernel failed: %+v \n", err)
		return
	}

	if err := queue.Finish(); err != nil {
		fmt.Printf("Finish failed: %+v \n", err)
		return
	}

	inverse_input := make([]float32, len(data))

	if _, err := queue.EnqueueReadBufferFloat32(rev_in, true, 0, inverse_input, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	fmt.Printf("cs1= %25.24g\n", float64(math.Cos(2.0*math.Pi/7.0)))
	fmt.Printf("sn1= %25.24g\n", float64(math.Sin(2.0*math.Pi/7.0)))
	fmt.Printf("cs2= %25.24g\n", float64(math.Cos(4.0*math.Pi/7.0)))
	fmt.Printf("sn2= %25.24g\n", float64(math.Sin(4.0*math.Pi/7.0)))
	fmt.Printf("cs3= %25.24g\n", float64(math.Cos(6.0*math.Pi/7.0)))
	fmt.Printf("sn3= %25.24g\n", float64(math.Sin(6.0*math.Pi/7.0)))
	fmt.Printf("Over 7 = %25.24g\n", float64(1.0/7.0))

	// Print data input
	for i := 0; i < nElem; i++ {
		fmt.Printf("Data In: (")
		for j := 0; j < 7; j++ {
			fmt.Printf("%f + i*%f", data[2*i+j*2*nElem], data[2*i+j*2*nElem+1])
			if j == 6 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print golden result
	for i := 0; i < nElem; i++ {
		fmt.Printf("Golden Result: (")
		for j := 0; j < 7; j++ {
			fmt.Printf("%f + i*%f", gold_res[2*i+j*2*nElem], gold_res[2*i+j*2*nElem+1])
			if j == 6 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print returned result
	for i := 0; i < nElem; i++ {
		fmt.Printf("Returned FFT7: (")
		for j := 0; j < 7; j++ {
			fmt.Printf("%f + i*%f", results[2*i+j*2*nElem], results[2*i+j*2*nElem+1])
			if j == 6 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	// Print inversed FFT
	for i := 0; i < nElem; i++ {
		fmt.Printf("Inverse FFT7: (")
		for j := 0; j < 7; j++ {
			fmt.Printf("%f + i*%f", inverse_input[2*i+j*2*nElem], inverse_input[2*i+j*2*nElem+1])
			if j == 6 {
			} else {
				fmt.Printf("; ")
			}
		}
		fmt.Printf(")\n")
	}
	
	correct := 0
	max_relerr, max_abserr := float64(-1e-6), float64(-1e-6)
	for i, v := range inverse_input {
		if gold_ret[i] == v {
			correct++
		} else {
			if gold_ret[i] != 0 {
				tmp := (v - gold_ret[i]) / gold_ret[i]
				tmp1 := math.Abs(float64(tmp))
				if tmp1 < 1e-6 {
					correct++
				} else {
					if tmp1 > max_relerr {
						max_relerr = tmp1
					}
				}
			} else {
				tmp2 := math.Abs(float64(v))
				if tmp2 < 1e-6 {
					correct++
				} else {
					if tmp2 > max_abserr {
						max_abserr = tmp2
					}
				}
			}
		}
	}

	if correct != len(data) {
		fmt.Printf("%d/%d correct values \n", correct, len(data))
		fmt.Printf("Max. rel. error: %g\n", max_relerr)
		fmt.Printf("Max. abs. error: %g\n", max_abserr)
		return
	}

	fmt.Printf("Finished tests on fft7_c2c_long_interleaved_oop\n")

	fmt.Printf("freeing resources \n")
	input.Release()
	output.Release()
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}
