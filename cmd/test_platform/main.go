// mumax3 main command
package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	//	"github.com/mumax/3cl/util"
	"log"
	//	"os"
	"runtime"
	//	"time"
)

var (
	//	flag_cachedir      = flag.String("cache", "", "Kernel cache directory")
	//	flag_cpuprof       = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	//	flag_failfast      = flag.Bool("failfast", false, "If one simulation fails, stop entire batch immediately")
	//	flag_forceclean    = flag.Bool("f", true, "Force start, clean existing output directory")
	//	flag_gpu           = flag.Int("gpu", 0, "Specify GPU")
	//	flag_platform      = flag.Int("platform", 0, "Specify OpenCL platform")
	//	flag_testkern	   = flag.Bool("kerns", false, "Test list of kernels created versus programmed list of kernels")
	//	flag_interactive   = flag.Bool("i", false, "Open interactive browser session")
	//	flag_launchtimeout = flag.Duration("launchtimeout", 0, "Launch timeout for CUDA calls")
	//	flag_memprof       = flag.Bool("memprof", false, "Recored gopprof memory profile")
	//	flag_od            = flag.String("o", "", "Override output directory")
	//	flag_port          = flag.String("http", ":35367", "Port to serve web gui")
	//	flag_selftest      = flag.Bool("paranoid", false, "Enable convolution self-test for cuFFT sanity.")
	//	flag_silent        = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	//	flag_sync          = flag.Bool("sync", false, "Synchronize all CUDA calls (debug)")
	//	flag_test          = flag.Bool("test", false, "Cuda test (internal)")
	flag_version = flag.Bool("v", true, "Print version")

//	flag_vet           = flag.Bool("vet", false, "Check input files for errors, but don't run them")
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	platforms, err := cl.GetPlatforms()
	if err != nil {
		fmt.Printf("Failed to get platforms: %+v \n", err)
	}
	for i, p := range platforms {
		fmt.Printf("Platform %d: \n", i)
		fmt.Printf("  Name: %s \n", p.Name())
		fmt.Printf("  Vendor: %s \n", p.Vendor())
		fmt.Printf("  Profile: %s \n", p.Profile())
		fmt.Printf("  Version: %s \n", p.Version())
		fmt.Printf("  Extensions: %s \n", p.Extensions())
	}
	platform := platforms[0]

	devices, err := platform.GetDevices(cl.DeviceTypeAll)
	if err != nil {
		fmt.Printf("Failed to get devices: %+v \n", err)
		return
	}
	if len(devices) == 0 {
		fmt.Printf("GetDevices returned no devices \n")
	}
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

	runtime.GOMAXPROCS(runtime.NumCPU())
}

// print version to stdout
func printVersion() {
	//	fmt.Print("//", engine.UNAME, "\n")
	fmt.Print(opencl.PlatformInfo)
	fmt.Print("//", opencl.GPUInfo)
	fmt.Print("//(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
	fmt.Print("//This is free software without any warranty. See license.txt", "\n")
}
