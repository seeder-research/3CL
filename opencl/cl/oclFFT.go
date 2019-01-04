/*
Package extends clFFT with Bluesteins algorithm to enable FFT of any radix
*/

package cl

//////// Wrapper to allow function pointers ////////
type OclFFTFuncs interface {
	Exec(dst, src *MemObject)
}

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
	inStride      int
	outStride     int
	inDist        int
	outDist       int
	clfftplans    [3]*ClFFTPlan
	buffers       [3]*MemObject
	exec_sequence []OclFFTFuncs
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

func (p *OclFFTPlan) GetDimension() ClFFTDimension {
	return p.dimension
}

func (p *OclFFTPlan) GetResultLocation() ClFFTResultLocation {
	return p.placeness
}

func (p *OclFFTPlan) GetInStride() int {
	return p.inStride
}

func (p *OclFFTPlan) GetOutStride() int {
	return p.outStride
}

func (p *OclFFTPlan) GetInDistance() int {
	return p.inDist
}

func (p *OclFFTPlan) GetOutDistance() int {
	return p.outDist
}

func (p *OclFFTPlan) GetBatchCount() int {
	return p.batch
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

func (p *OclFFTPlan) SetDimension(in ClFFTDimension) {
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

func (p *OclFFTPlan) SetInStride(in int) {
	if p.inStride != in {
		p.inStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutStride(in int) {
	if p.outStride != in {
		p.outStride = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetInDist(in int) {
	if p.inDist != in {
		p.inDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetOutDist(in int) {
	if p.outDist != in {
		p.OutDist = in
		p.bake = false
	}
}

func (p *OclFFTPlan) SetBatchCount(in int) {
	if p.batch != in {
		p.batch = in
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

	for funcs, _ := range p.exec_sequence {
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
