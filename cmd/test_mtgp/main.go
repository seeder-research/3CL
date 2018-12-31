package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	"math/rand"
	"os"
	"time"
	"unsafe"
)

var d_length = flag.Int("size", 1024, "Total number of random numbers to generate")
var r_seed = flag.Uint("seed", 0, "Seed value of RNG")
var d_dump = flag.Bool("dump", false, "Whether to dump generated values to screen")
var gpu_flag = flag.Int("gpu", 0, "GPU selected for operation")

func main() {

	flag.Parse()

	opencl.Init(*gpu_flag)
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

	context, queue, kernels := opencl.ClCtx, opencl.ClCmdQueue, opencl.KernList

	fmt.Printf("Initializing MTGP RNG and generate uniformly distributed numbers... \n")

	seed := InitRNG()
	fmt.Println("Seed: ", seed)
	rng := opencl.NewGenerator("mtgp")
	rng.PRNG.Init(seed, nil)

	fmt.Printf("Creating output buffer... \n")
	d_size := int(*d_length)
	data := make([]float32, d_size)
	for i := 0; i < d_size; i++ {
		data[i] = float32(53.0)
	}
	output, err := context.CreateEmptyBuffer(cl.MemReadWrite, 4*len(data))
	if err != nil {
		fmt.Printf("CreateBuffer failed for output: %+v \n", err)
		return
	}
	if _, err := queue.EnqueueWriteBufferFloat32(output, true, 0, data, nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	if *d_dump {
		fmt.Println("Results before execution: ", results)
	}

	event := rng.PRNG.GenerateUniform((unsafe.Pointer)(output), d_size, nil)
	err = cl.WaitForEvents([]*cl.Event{event})
	if err != nil {
		fmt.Printf("WaitForEvents failed for GenerateUniform: %+v \n", err)
		return
	}

	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	if *d_dump {
		fmt.Println("Results after execution: ", results)
	}

	fOut, fErr := os.Create("uniform_bytes.bin")
	if fErr != nil {
		panic(fErr)
	}

	wr := bufio.NewWriter(fOut)
	outBytes := new(bytes.Buffer)
	for _, v := range results {
		vErr := binary.Write(outBytes, binary.LittleEndian, v)
		if vErr != nil {
			fmt.Println("binary.Write failed: ", vErr)
		}
	}

	nn, wrErr := wr.Write(outBytes.Bytes())
	if wrErr != nil {
		fmt.Println("bufio.Write failed: ", wrErr)
	} else {
		wr.Flush()
		fmt.Println("Wrote ", nn, "bytes to file")
	}
	fOut.Close()

	fmt.Printf("Re-initializing MTGP RNG and generate normally distributed numbers... \n")

	rng.PRNG.Init(seed, nil)

	if _, err := queue.EnqueueWriteBufferFloat32(output, true, 0, data, nil); err != nil {
		fmt.Printf("EnqueueWriteBufferFloat32 failed: %+v \n", err)
		return
	}

	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}

	if *d_dump {
		fmt.Println("Results before execution: ", results)
	}

	event = rng.PRNG.GenerateNormal((unsafe.Pointer)(output), d_size, nil)
	err = cl.WaitForEvents([]*cl.Event{event})
	if err != nil {
		fmt.Printf("WaitForEvents failed for GenerateNormal: %+v \n", err)
		return
	}

	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		fmt.Printf("EnqueueReadBufferFloat32 failed: %+v \n", err)
		return
	}
	if *d_dump {
		fmt.Println("Results after execution: ", results)
	}

	fOut, fErr = os.Create("norm_bytes.bin")
	if fErr != nil {
		panic(fErr)
	}

	wr = bufio.NewWriter(fOut)
	outBytes = new(bytes.Buffer)
	for _, v := range results {
		vErr := binary.Write(outBytes, binary.LittleEndian, v)
		if vErr != nil {
			fmt.Println("binary.Write failed: ", vErr)
		}
	}

	nn, wrErr = wr.Write(outBytes.Bytes())
	if wrErr != nil {
		fmt.Println("bufio.Write failed: ", wrErr)
	} else {
		wr.Flush()
		fmt.Println("Wrote ", nn, "bytes to file")
	}

	fOut.Close()

	fmt.Printf("Finished tests on MTGP RNG\n")

	fmt.Printf("freeing resources \n")
	output.Release()
	for _, krn := range kernels {
		krn.Release()
	}

	opencl.ReleaseAndClean()
}

func InitRNG() uint32 {
	if *r_seed == (uint)(0) {
		rand.Seed(time.Now().UTC().UnixNano())
		return rand.Uint32()
	}
	return (uint32)(*r_seed)
}
