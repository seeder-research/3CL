package opencl

// INTERNAL
// Base implementation for all FFT plans.

import (
	"github.com/mumax/3cl/opencl/cl"
)

// Base implementation for all FFT plans.
type fftplan struct {
	handle *cl.ClFFTPlan
}

func prod3(x, y, z int) int {
	return x * y * z
}

// Releases all resources associated with the FFT plan.
func (p *fftplan) Free() {
	if p.handle != nil {
		p.handle.Destroy()
		p.handle = nil
	}
}

func calcChirpLength(x int) int {
	if (x < 1) {
		return -1
	}
	tmp := x
	fft_primes := []int{13, 11, 8, 7, 5, 4, 3, 2}
	for _, vv := range fft_primes {
		for tmp % vv == 0 {
			tmp /= vv
		}
	}
	return tmp
}

func chkFFTSize(x, y, z int) (bool, [3]int) {
	tmp := [3]int{calcChirpLength(x), calcChirpLength(y), calcChirpLength(z)}
	flag := true
	if tmp[0]*tmp[1]*tmp[2] != 1 {
		flag = false
	}
	return flag, tmp
}
