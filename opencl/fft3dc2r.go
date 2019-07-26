package opencl

import (
	"fmt"
	"log"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/timer"
)

// 3D single-precision real-to-complex FFT plan.
type fft3DC2RPlan struct {
	fftplan
	size [3]int
}

// 3D single-precision real-to-complex FFT plan.
func newFFT3DC2R(Nx, Ny, Nz int) fft3DC2RPlan {
        // TODO: Update plan creation to create plan from new FFT library
	handle, err := cl.NewCLFFTPlan(ClCtx, cl.CLFFTDim3D, []int{Nx, Ny, Nz}) // new xyz swap
	if err != nil {
		log.Printf("Unable to create fft3dc2r plan \n")
	}
        // TODO: The next lines running Set<> functions configure
        // the clFFT plans for the simulator. These need to be
        // updated accordingly for the new FFT library
	arrLayout := cl.NewArrayLayout()
	arrLayout.SetInputLayout(cl.CLFFTLayoutHermitianInterleaved)
	arrLayout.SetOutputLayout(cl.CLFFTLayoutReal)
	err = handle.SetLayouts(arrLayout)
	if err != nil {
		log.Printf("Unable to set buffer layouts of fft3dc2r plan \n")
	}

	InStrideArr := []int{1, Nx/2 + 1, Ny * (Nx/2 + 1)}
	err = handle.SetInStride(InStrideArr)
	if err != nil {
		log.Printf("Unable to set input stride of fft3dc2r plan \n")
	}

	err = handle.SetResultOutOfPlace()
	if err != nil {
		log.Printf("Unable to set placeness of fft3dc2r result \n")
	}

	err = handle.SetSinglePrecision()
	if err != nil {
		log.Printf("Unable to set precision of fft3dc2r plan \n")
	}

	err = handle.SetResultNoTranspose()
	if err != nil {
		log.Printf("Unable to set transpose of fft3dc2r result \n")
	}

	err = handle.SetScale(cl.ClFFTDirectionBackward, float32(1.0))
	if err != nil {
		log.Printf("Unable to set scaling factor of fft3dc2r result \n")
	}

        // TODO: After configuring the plans, bakeplan is run so that
        // it does not need to be run when plan is first executed
	err = handle.BakePlanSimple([]*cl.CommandQueue{ClCmdQueue})
	if err != nil {
		log.Printf("Unable to bake fft3dc2r plan \n")
	}

	return fft3DC2RPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) ([]*cl.Event, error) {
	if Synchronous {
		ClCmdQueue.Finish()
		timer.Start("fft")
	}
	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLenFloats()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	tmpPtr := src.DevPtr(0)
	srcMemObj := *(*cl.MemObject)(tmpPtr)
	tmpPtr = dst.DevPtr(0)
	dstMemObj := *(*cl.MemObject)(tmpPtr)
        // TODO: the following line needs to be updated for the new FFT library
	eventsList, err := p.handle.EnqueueBackwardTransform([]*cl.CommandQueue{ClCmdQueue}, []*cl.Event{src.GetEvent(0), dst.GetEvent(0)},
		[]*cl.MemObject{&srcMemObj}, []*cl.MemObject{&dstMemObj}, nil)
	if Synchronous {
		ClCmdQueue.Finish()
		timer.Stop("fft")
	}
	return eventsList, err
}

// 3D size of the input array.
func (p *fft3DC2RPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return 2 * (p.size[X]/2 + 1), p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DC2RPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DC2RPlan) InputLenFloats() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DC2RPlan) OutputLenFloats() int {
	return prod3(p.OutputSizeFloats())
}
