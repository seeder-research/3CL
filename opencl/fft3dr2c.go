package opencl

import (
	"log"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/timer"
	"github.com/mumax/3cl/util"
)

// 3D single-precision real-to-complex FFT plan.
type fft3DR2CPlan struct {
	fftplan
	size [3]int
}

// 3D single-precision real-to-complex FFT plan.
func newFFT3DR2C(Nx, Ny, Nz int) fft3DR2CPlan {
        // TODO: Update plan creation to create plan from new FFT library
	handle, err := cl.NewCLFFTPlan(ClCtx, cl.CLFFTDim3D, []int{Nx, Ny, Nz})
	if err != nil {
		log.Printf("Unable to create fft3dr2c plan \n")
	}
        // TODO: The next lines running Set<> functions configure
        // the clFFT plans for the simulator. These need to be
        // updated accordingly for the new FFT library
	arrLayout := cl.NewArrayLayout()
	arrLayout.SetInputLayout(cl.CLFFTLayoutReal)
	arrLayout.SetOutputLayout(cl.CLFFTLayoutHermitianInterleaved)
	err = handle.SetLayouts(arrLayout)
	if err != nil {
		log.Printf("Unable to set buffer layouts of fft3dr2c plan \n")
	}

	OutStrideArr := []int{1, Nx/2 + 1, Ny * (Nx/2 + 1)}
	err = handle.SetOutStride(OutStrideArr)
	if err != nil {
		log.Printf("Unable to set output stride of fft3dr2c plan \n")
	}

	err = handle.SetResultOutOfPlace()
	if err != nil {
		log.Printf("Unable to set placeness of fft3dr2c result \n")
	}

	err = handle.SetSinglePrecision()
	if err != nil {
		log.Printf("Unable to set precision of fft3dr2c plan \n")
	}

	err = handle.SetResultNoTranspose()
	if err != nil {
		log.Printf("Unable to set transpose of fft3dr2c result \n")
	}

        // TODO: After configuring the plans, bakeplan is run so that
        // it does not need to be run when plan is first executed
	err = handle.BakePlanSimple([]*cl.CommandQueue{ClCmdQueue})
	if err != nil {
		log.Printf("Unable to bake fft3dr2c plan \n")
	}
	return fft3DR2CPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DR2CPlan) ExecAsync(src, dst *data.Slice) ([]*cl.Event, error) {
	if Synchronous {
		ClCmdQueue.Finish()
		timer.Start("fft")
	}
	util.Argument(src.NComp() == 1 && dst.NComp() == 1)
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		log.Panicf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len())
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		log.Panicf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len())
	}
	tmpPtr := src.DevPtr(0)
	srcMemObj := *(*cl.MemObject)(tmpPtr)
	tmpPtr = dst.DevPtr(0)
	dstMemObj := *(*cl.MemObject)(tmpPtr)
        // TODO: the following line needs to be updated for the new FFT library
 	eventsList, err := p.handle.EnqueueForwardTransform([]*cl.CommandQueue{ClCmdQueue}, []*cl.Event{src.GetEvent(0), dst.GetEvent(0)},
		[]*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
	if Synchronous {
		ClCmdQueue.Finish()
		timer.Stop("fft")
	}
	return eventsList, err
}

// 3D size of the input array.
func (p *fft3DR2CPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DR2CPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return 2 * (p.size[X]/2 + 1), p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DR2CPlan) InputLen() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DR2CPlan) OutputLen() int {
	return prod3(p.OutputSizeFloats())
}
