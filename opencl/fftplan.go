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
