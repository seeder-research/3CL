/*
Package extends clFFT with Bluesteins algorithm to enable FFT of any radix
*/

package cl

import (
	"fmt"
)

//////// Wrapper to allow function pointers ////////
type OclFFTFuncs struct {
	ExecFunc func(dst, src *MemObject) error
}

func (p *OclFFTFuncs) Exec(dst, src *MemObject) error {
	return p.ExecFunc(dst, src)
}

//////// Map data structure for storing kernels in OclFFTPlan struct ////////
type KernelMap map[string]*Kernel

//////// Data structure for twiddle buffers ////////
type chirpArray []int

//////// Map data structure for storing twiddle buffers for Bluesteins ////////
type forwardChirpTwiddles map[int]*chirpArray
type forwardChirpTwiddlesFFT map[int]*chirpArray

//////// Map data structure for storing twiddle buffers for Bluesteins ////////
type backwardChirpTwiddles map[int]*chirpArray
type backwardChirpTwiddlesFFT map[int]*chirpArray

//////// Radices and maximum length supported by clFFT ////////
var supported_radices = []int{17, 13, 11, 8, 7, 5, 4, 3, 2}

const maxLen int = 128000000

//////// Wrapper plan to interface with clFFT ////////
type OclFFTPlan struct {
	bake          bool
	direction     ClFFTDirection
	precision     ClFFTPrecision
	layout        ClFFTLayout
	dimension     ClFFTDim
	placeness     ClFFTResultLocation
	chirpz        [3]bool
	chirp_lengths [3]int
	fftLengths    [3]int
	batches       int
	inStride      [3]int
	outStride     [3]int
	inDist        int
	outDist       int
	clfftplans    [3]*ClFFTPlan
	buffers       [3]*MemObject
	exec_sequence []*OclFFTFuncs
	clCtx         *Context
	clDevice      *Device
	clKernels     KernelMap
	clProg        *Program
}

//////// private library functions ////////
func createOclFFTPlan() *OclFFTPlan {
	newPlan := new(OclFFTPlan)
	return newPlan
}

func determineLengthCompatibility(in int) bool {
	var outFlag bool
	outFlag = false
	if in < 1 || in > maxLen {
		return outFlag
	}

	result := in
	result_length := 0
	for _, rad := range supported_radices {
		result_length = result % rad
		for result_length == 0 {
			result /= rad
			result_length = result % rad
		}
	}
	if result == 1 {
		return true
	}
	return false
}

func determineChirpLength(in int) int {
	outLength := -1
	if in < 1 || in > maxLen/2 {
		return outLength
	}

	outLength = 2 * in
	for determineLengthCompatibility(outLength) == false {
		outLength += 1
	}
	return outLength
}

func (p *OclFFTPlan) setChirp() {
	for ind, dimLen := range p.fftLengths {
		if dimLen > 1 {
			chkFlag := determineLengthCompatibility(dimLen)
			if chkFlag {
				p.chirpz[ind] = false
				p.chirp_lengths[ind] = -1
			} else {
				p.chirpz[ind] = true
				p.chirp_lengths[ind] = determineChirpLength(dimLen)
			}
		}
	}
}

//////// public library functions ////////
func CreateDefaultOclFFTPlan() (*OclFFTPlan, error) {
	newPlan := createOclFFTPlan()
	newPlan.SetContext(nil)
	return newPlan, nil
}

func (p *OclFFTPlan) GetContext() *Context {
	return p.clCtx
}

func (p *OclFFTPlan) GetDevice() *Device {
	return p.clDevice
}

func (p *OclFFTPlan) GetProgram() *Program {
	return p.clProg
}

func (p *OclFFTPlan) GetKernel(in string) *Kernel {
	return p.clKernels[in]
}

func (p *OclFFTPlan) GetDirection() ClFFTDirection {
	return p.direction
}

func (p *OclFFTPlan) GetPrecision() ClFFTPrecision {
	return p.precision
}

func (p *OclFFTPlan) GetLayout() ClFFTLayout {
	return p.layout
}

func (p *OclFFTPlan) GetDimension() ClFFTDim {
	return p.dimension
}

func (p *OclFFTPlan) GetResultLocation() ClFFTResultLocation {
	return p.placeness
}

func (p *OclFFTPlan) GetLengths() [3]int {
	return p.fftLengths
}

func (p *OclFFTPlan) GetInStride() [3]int {
	return p.inStride
}

func (p *OclFFTPlan) GetOutStride() [3]int {
	return p.outStride
}

func (p *OclFFTPlan) GetInDistance() int {
	return p.inDist
}

func (p *OclFFTPlan) GetOutDistance() int {
	return p.outDist
}

func (p *OclFFTPlan) GetBatchCount() int {
	return p.batches
}

func (p *OclFFTPlan) SetContext(in *Context) {
	if p.clCtx == nil || p.clCtx != in {
		p.clCtx = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetDevice(in *Device) {
	if p.clDevice == nil || p.clDevice != in {
		p.clDevice = in
		p.bake = false
	}
}

func (p *OclFFTPlan) setProgram() error {
	if p.clCtx == nil || p.clDevice == nil {
		return toError(nil)
	}
	var err error
	p.clProg, err = createOclFFTProgram(p.clCtx)
	if err != nil {
		fmt.Printf("CreateProgramWithSource failed: %+v \n", err)
		return err
	}
	if err = p.clProg.BuildProgram([]*Device{p.clDevice}, "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -cl-kernel-arg-info"); err != nil {
		fmt.Printf("BuildProgram failed: %+v \n", err)
		return err
	}
	for _, kernname := range KernelNames {
		p.clKernels[kernname], err = p.clProg.CreateKernel(kernname)
		if err != nil {
			fmt.Printf("CreateKernel failed: %+v \n", err)
			return err
		}
	}
	return toError(nil)
}

func (p *OclFFTPlan) SetDirection(in ClFFTDirection) {
	if p.direction != in {
		p.direction = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetPrecision(in ClFFTPrecision) {
	if p.precision != in {
		p.precision = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetLayout(in ClFFTLayout) {
	if p.layout != in {
		p.layout = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetDimension(in ClFFTDim) {
	if p.dimension != in {
		p.dimension = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetResultLocation(in ClFFTResultLocation) {
	if p.placeness != in {
		p.placeness = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetLengths(in [3]int) {
	if p.fftLengths != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.fftLengths = in
		p.bake = false
		p.setChirp()
	}
}

func (p *OclFFTPlan) SetInStride(in [3]int) {
	if p.inStride != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.inStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutStride(in [3]int) {
	if p.outStride != in && in[0] > 0 && in[1] > 0 && in[2] > 0 {
		p.outStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetInDist(in int) {
	if p.inDist != in && in > 0 {
		p.inDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutDist(in int) {
	if p.outDist != in && in > 0 {
		p.outDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetBatchCount(in int) {
	if p.batches != in && in > 0 {
		p.batches = in
		p.bake = false
	}
}

func (p *OclFFTPlan) Destroy() {
}

func (p *OclFFTPlan) Bake() error {
	p.bake = true
	return toError(nil)
}

func (p *OclFFTPlan) ExecTransform(dst, src *MemObject) error {
	if p.bake != true {
		test := p.Bake()
		if test != nil {
			return toError(test)
		}
	}

	for _, funcs := range p.exec_sequence {
		test := funcs.Exec(dst, src)
		if test != nil {
			return toError(test)
		}
	}

	return toError(nil)
}

func OclFFTTearDown() error {
	return toError(nil)
}
