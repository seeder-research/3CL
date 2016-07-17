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
	flag_gpu           = flag.Int("gpu", 0, "Specify GPU")
	flag_platform      = flag.Int("platform", 0, "Specify OpenCL platform")
	flag_testkern	   = flag.Bool("kerns", false, "Test list of kernels created versus programmed list of kernels")
//	flag_interactive   = flag.Bool("i", false, "Open interactive browser session")
//	flag_launchtimeout = flag.Duration("launchtimeout", 0, "Launch timeout for CUDA calls")
//	flag_memprof       = flag.Bool("memprof", false, "Recored gopprof memory profile")
//	flag_od            = flag.String("o", "", "Override output directory")
//	flag_port          = flag.String("http", ":35367", "Port to serve web gui")
//	flag_selftest      = flag.Bool("paranoid", false, "Enable convolution self-test for cuFFT sanity.")
//	flag_silent        = flag.Bool("s", false, "Silent") // provided for backwards compatibility
//	flag_sync          = flag.Bool("sync", false, "Synchronize all CUDA calls (debug)")
//	flag_test          = flag.Bool("test", false, "Cuda test (internal)")
	flag_version       = flag.Bool("v", true, "Print version")
//	flag_vet           = flag.Bool("vet", false, "Check input files for errors, but don't run them")
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	opencl.Init(*flag_gpu,*flag_platform)
	runtime.GOMAXPROCS(runtime.NumCPU())
//	cuda.Synchronous = *flag_sync
	if *flag_version {
		printVersion()
	}

	if *flag_testkern {
		TestKernels()
	}
}

// print version to stdout
func printVersion() {
//	fmt.Print("//", engine.UNAME, "\n")
	fmt.Print(opencl.PlatformInfo)
	fmt.Print("//", opencl.GPUInfo)
	fmt.Print("//(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
	fmt.Print("//This is free software without any warranty. See license.txt", "\n")
}

func TestKernels() {
	if len(opencl.KernList) == 0 {
		fmt.Println("No kernels found!")
		return
	}
	for kernname, _ := range opencl.KernList {
		fmt.Printf("Found kernel: %s \n", kernname)
		PrintKernelInfo(kernname)
	}
}

func PrintKernelInfo(name string) {
	totalArgs, err := opencl.KernList[name].NumArgs()
	if err != nil {
		fmt.Printf("Failed to get number of arguments of kernel: $+v \n", err)
	} else {
		fmt.Printf("Number of arguments in kernel : %d \n", totalArgs)
	}
	for i := 0; i < totalArgs; i++ {
		name, err := opencl.KernList[name].ArgName(i)
		if err == cl.ErrUnsupported {
			break
		} else if err != nil {
			fmt.Printf("GetKernelArgInfo for name failed: %+v \n", err)
			break
		} else {
			fmt.Printf("Kernel arg %d: %s \n", i, name)
		}
	}
}
