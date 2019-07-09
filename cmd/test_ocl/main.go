package main

import (
	//"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"

	//"github.com/mumax/3cl/cmd/test_blu2d/purefft"

	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/opencl/cl"
	//"github.com/mumax/3cl/cmd/test_blu_2d"
)

var (
	Flag_gpu   = flag.Int("gpu", 0, "Specify GPU")
	Flag_size  = flag.Int("length", 17, "length of data to test")
	Flag_print = flag.Bool("print", false, "Print out result")
	Flag_comp  = flag.Int("components", 1, "Number of components to test")
	//Flag_conj  = flag.Bool("conjugate", false, "Conjugate B in multiplication")
)

//////// Radices and maximum length supported by clFFT ////////
//var supported_radices = []int{17, 13, 11, 8, 7, 5, 4, 3, 2}
var supported_radices = []int{13, 11, 8, 7, 5, 4, 3, 2}

const maxLen int = 128000000

//HermitianWarning Issue a warning if complex conjugates of hermitian are not closely matching

func main() {

	flag.Parse()
	//var Desci int //Descision variable
	N := int(*Flag_size)
	opencl.Init(*Flag_gpu)
	//rand.Seed(time.Now().Unix())
	rand.Seed(24)
	//X := make([]float32, 2*N)
	NComponents := int(*Flag_comp)
	if N < 4 {
		fmt.Println("argument to -fft must be 4 or greater!")
		os.Exit(-1)
	}
	if (NComponents < 1) || (NComponents > 3) {
		fmt.Println("argument to -components must be 1, 2 or 3!")
		os.Exit(-1)
	}

	//opencl.Init(*engine.Flag_gpu)

	/* Print input array */

	//plan1d := FftPlanValue{false, true, true, false, N, 1, 1, 0}

	/* Prepare OpenCL memory objects and place data inside them for . */
	//Initialize GPU with a flag to pick the desired gpu
	//opencl.Init(*engine.Flag_gpu)

	platform := opencl.ClPlatform
	fmt.Printf("Platform in use: \n")
	fmt.Printf("  Vendor: %s \n", platform.Vendor())
	fmt.Printf("  Profile: %s \n", platform.Profile())
	fmt.Printf("  Version: %s \n", platform.Version())
	fmt.Printf("  Extensions: %s \n", platform.Extensions())

	fmt.Printf("Device in use: \n")

	d := opencl.ClDevice
	//fmt.Printf("Device %d (%s): %s \n", *engine.Flag_gpu, d.Type(), d.Name())
	fmt.Printf("  Address Bits: %d \n", d.AddressBits())

	//queue := opencl.ClCmdQueue

	effort, _ := cl.CreateDefaultOclFFTPlan()
	effort.SetLengths([3]int{8, 1, 1})
	fmt.Printf("Printing array \n")
	fmt.Printf("%v", effort.GetLengths())

	// fmt.Printf("\n Executing Forward 2D FFT. Printing input array \n")
	// plan2d := FftPlan2DValue{false, true, true, true, false, int(*Flag_size), 2, 1, int(*Flag_size), 2}
	// inputs2d := make([][]float32, NComponents)
	// var size2d [3]int

	// if plan2d.IsForw && plan2d.IsRealHerm {
	// 	size2d = [3]int{plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs2d[i] = make([]float32, size2d[0])
	// 		for j := 0; j < plan2d.ColDim; j++ {
	// 			for k := 0; k < plan2d.RowDim; k++ {
	// 				inputs2d[i][j*plan2d.RowDim+k] = float32(j*plan2d.RowDim+k) * float32(0.1) //float32(0.1)
	// 				fmt.Printf("( %f ) ", inputs2d[i][j*plan2d.RowDim+k])
	// 			}
	// 			fmt.Printf("\n")
	// 		}
	// 	}
	// }

	// if plan2d.IsForw && !plan2d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size2d = [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs2d[i] = make([]float32, size2d[0])
	// 		for j := 0; j < plan2d.ColDim; j++ {
	// 			for k := 0; k < plan2d.RowDim; k++ {
	// 				inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim+k) * float32(0.1)
	// 				inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim+k) * float32(0.1)
	// 				fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// 			}
	// 			//fmt.Printf("\n")
	// 		}
	// 	}
	// }

	// if !plan2d.IsForw && plan2d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size2d = [3]int{2 * int(1+plan2d.RowDim/2) * plan2d.ColDim, 1, 1}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < int(1+plan2d.RowDim/2); k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*int(1+plan2d.RowDim/2) + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*int(1+plan2d.RowDim/2) + k) //float32(0.1)
	// 	// 			//fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)], inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)+1])
	// 	// 		}
	// 	// 		//fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// 	inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 		-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 		-1.700001, 0.157525,
	// 		-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 		-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 		0.000000, 0.000000}
	// }

	// if !plan2d.IsRealHerm && !plan2d.IsForw {
	// 	size2d = [3]int{2 * plan2d.RowDim * plan2d.ColDim, 1, 1}
	// 	inputs2d[0] = []float32{56.099998, 0.000001, -1.700004, 9.094196, -1.700002, 4.388192, -1.699986, 2.745589,
	// 		-1.699995, 1.864812, -1.700000, 1.283777, -1.699999, 0.846498, -1.700004, 0.483691,
	// 		-1.700001, 0.157525, -1.700001, -0.157525, -1.700004, -0.483691, -1.699999, -0.846498,
	// 		-1.700000, -1.283777, -1.699995, -1.864812, -1.699986, -2.745589, -1.700002, -4.388192,
	// 		-1.700004, -9.094196,
	// 		-28.900002, 0.000000, 0.000001, -0.000000, 0.000001, 0.000006, -0.000006, 0.000001,
	// 		-0.000002, 0.000000, -0.000000, 0.000002, -0.000001, 0.000001, 0.000002, 0.000000,
	// 		0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000, -0.000001, -0.000001,
	// 		-0.000000, -0.000002, -0.000002, 0.000000, -0.000006, -0.000001, 0.000001, -0.000006,
	// 		0.000001, 0.000000}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < plan2d.RowDim; k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim + k) //float32(0.1)
	// 	// 			fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// 	// 		}
	// 	// 		fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// }

	// fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	// cpuArray2d := data.SliceFromArray(inputs2d, size2d)
	// gpu2dBuffer := opencl.Buffer(NComponents, size2d)
	// //outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	// data.Copy(gpu2dBuffer, cpuArray2d)

	// fmt.Println("Waiting for data transfer to complete...")
	// queue.Finish()
	// fmt.Println("Input data transfer completed.")

	// //opencl.Recycle(gpu2dBuffer)

	// Parse2D(gpu2dBuffer, plan2d)

	// fmt.Printf("\n Executing 3D FFT \n")

	// plan3d := FftPlan3DValue{false, true, true, true, false, true, int(*Flag_size), 2, 2, int(*Flag_size), 2, 2}
	// inputs3d := make([][]float32, NComponents)
	// var size3d [3]int

	// if plan3d.IsForw && plan3d.IsRealHerm {
	// 	size3d = [3]int{plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs3d[i] = make([]float32, size3d[0])
	// 		for z := 0; z < plan3d.DepthDim; z++ {
	// 			for j := 0; j < plan3d.ColDim; j++ {
	// 				for k := 0; k < plan3d.RowDim; k++ {
	// 					inputs3d[i][z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01) //float32(0.1)
	// 					fmt.Printf("( %f ) ", inputs3d[i][z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k])
	// 				}
	// 				fmt.Printf("\n")
	// 			}
	// 		}
	// 	}
	// }

	// if plan3d.IsForw && !plan3d.IsRealHerm {
	// 	size3d = [3]int{2 * plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	for i := 0; i < NComponents; i++ {
	// 		inputs3d[i] = make([]float32, size3d[0])
	// 		for z := 0; z < plan3d.DepthDim; z++ {
	// 			for j := 0; j < plan3d.ColDim; j++ {
	// 				for k := 0; k < plan3d.RowDim; k++ {
	// 					inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01) //float32(0.1)
	// 					inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)+1] = float32(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k) * float32(0.01)
	// 					fmt.Printf("( %f , %f ) ", inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)], inputs3d[i][2*(z*plan3d.RowDim*plan3d.ColDim+j*plan3d.RowDim+k)+1])
	// 				}
	// 				fmt.Printf("\n")
	// 			}
	// 		}
	// 	}
	// }

	// if !plan3d.IsForw && plan3d.IsRealHerm {
	// 	fmt.Printf("\n Printing default value of hermitian matrix of 17*2 of FFT of 0,1,2,...,16;17,18,...,33")
	// 	size3d = [3]int{2 * int(1+plan3d.RowDim/2) * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	// for i := 0; i < NComponents; i++ {
	// 	// 	inputs2d[i] = make([]float32, size2d[0])
	// 	// 	for j := 0; j < plan2d.ColDim; j++ {
	// 	// 		for k := 0; k < int(1+plan2d.RowDim/2); k++ {
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*int(1+plan2d.RowDim/2) + k)   //float32(0.1)
	// 	// 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*int(1+plan2d.RowDim/2) + k) //float32(0.1)
	// 	// 			//fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)], inputs2d[i][2*(j*int(1+plan2d.RowDim/2)+k)+1])
	// 	// 		}
	// 	// 		//fmt.Printf("\n")
	// 	// 	}
	// 	// }
	// 	inputs3d[0] = []float32{22.7800, 0.0000, -0.3400, 1.8188, -0.3400, 0.8776, -0.3400, 0.5491, -0.3400, 0.3730, -0.3400, 0.2568, -0.3400, 0.1693, -0.3400, 0.0967, -0.3400, 0.0315,
	// 		-5.7800, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, -0.0000, -0.0000,
	// 		-11.5600, 0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000,
	// 		0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000}
	// }

	// if !plan3d.IsRealHerm && !plan3d.IsForw {
	// 	size3d = [3]int{2 * plan3d.RowDim * plan3d.ColDim * plan3d.DepthDim, 1, 1}
	// 	inputs3d[0] = []float32{22.7800, 22.7800, -2.1588, 1.4788, -1.2176, 0.5376, -0.8891, 0.2091, -0.7130, 0.0330, -0.5968, -0.0832, -0.5093, -0.1707, -0.4367, -0.2433,
	// 		-0.3715, -0.3085, -0.3085, -0.3715, -0.2433, -0.4367, -0.1707, -0.5093, -0.0832, -0.5968, 0.0330, -0.7130, 0.2091, -0.8891, 0.5376, -1.2176,
	// 		1.4788, -2.1588, -5.7800, -5.7800, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-11.5600, -11.5600, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000,
	// 		0.0000, 0.0000}

	// }
	// // for i := 0; i < NComponents; i++ {
	// // 	inputs2d[i] = make([]float32, size2d[0])
	// // 	for j := 0; j < plan2d.ColDim; j++ {
	// // 		for k := 0; k < plan2d.RowDim; k++ {
	// // 			inputs2d[i][2*(j*plan2d.RowDim+k)] = float32(j*plan2d.RowDim + k)   //float32(0.1)
	// // 			inputs2d[i][2*(j*plan2d.RowDim+k)+1] = float32(j*plan2d.RowDim + k) //float32(0.1)
	// // 			fmt.Printf(" (%f, %f) ", inputs2d[i][2*(j*plan2d.RowDim+k)], inputs2d[i][2*(j*plan2d.RowDim+k)+1])
	// // 		}
	// // 		fmt.Printf("\n")
	// // 	}
	// // }
	// // }

	// fmt.Println("\n Done. Transferring input data from CPU to GPU...")
	// cpuArray3d := data.SliceFromArray(inputs3d, size3d)
	// gpu3dBuffer := opencl.Buffer(NComponents, size3d)
	// //outBuffer := opencl.Buffer(NComponents, [3]int{2 * N, 1, 1})

	// data.Copy(gpu3dBuffer, cpuArray3d)

	// fmt.Println("Waiting for data transfer to complete...")
	// queue.Finish()
	// fmt.Println("Input data transfer completed.")

	// //Parse3D(gpu3dBuffer, plan3d)

	fmt.Printf("\n Finishing FFT......\n")
	opencl.ReleaseAndClean()
}
