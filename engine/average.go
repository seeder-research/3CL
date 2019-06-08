package engine

// Averaging of quantities over entire universe or just magnet.

import (
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
        "fmt"
)

// average of quantity over universe
func qAverageUniverse(q Quantity) []float64 {
	s := ValueOf(q)
	defer opencl.Recycle(s)
	return sAverageUniverse(s)
}

// average of slice over universe
func sAverageUniverse(s *data.Slice) []float64 {
        fmt.Printf("Over universe ... \n")
	nCell := float64(prod(s.Size()))
	avg := make([]float64, s.NComp())
	for i := range avg {
		avg[i] = float64(opencl.Sum(s.Comp(i))) / nCell
                fmt.Printf("Working on index: %d \n", i)
                fmt.Printf("nCell: %d \n", int(nCell))
                fmt.Printf("avg value: %+v \n", avg[i])
		checkNaN1(avg[i])
	}
	return avg
}

// average of slice over the magnet volume
func sAverageMagnet(s *data.Slice) []float64 {
        fmt.Printf("Averaging over magnet ... \n")
	if geometry.Gpu().IsNil() {
		return sAverageUniverse(s)
	} else {
		avg := make([]float64, s.NComp())
		for i := range avg {
			avg[i] = float64(opencl.Dot(s.Comp(i), geometry.Gpu())) / magnetNCell()
                        fmt.Printf("Working on index: %d \n", i)
			checkNaN1(avg[i])
		}
		return avg
	}
}

// number of cells in the magnet.
// not necessarily integer as cells can have fractional volume.
func magnetNCell() float64 {
	if geometry.Gpu().IsNil() {
		return float64(Mesh().NCell())
	} else {
		return float64(opencl.Sum(geometry.Gpu()))
	}
}
