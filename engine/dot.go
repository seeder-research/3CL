package engine

import (
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
)

func init() {
	DeclFunc("DotProduct", DotProduct, "Dot product of two vector quantities")
}

type dotProduct struct {
	a, b outputField
}

// DotProduct creates a new quantity that is the dot product of
// quantities a and b. E.g.:
// 	DotProct(&M, &B_ext)
func DotProduct(a, b outputField) *dotProduct {
	return &dotProduct{a, b}
}

func (d *dotProduct) Mesh() *data.Mesh {
	return d.a.Mesh()
}

func (d *dotProduct) NComp() int {
	return 1
}

func (d *dotProduct) Name() string {
	return d.a.Name() + "_dot_" + d.b.Name()
}

func (d *dotProduct) Unit() string {
	return d.a.Unit() + d.b.Unit()
}

func (d *dotProduct) Slice() (*data.Slice, bool) {
	slice := opencl.Buffer(d.NComp(), d.Mesh().Size())
	opencl.Zero(slice)
	A, r := d.a.Slice()
	if r {
		defer opencl.Recycle(A)
	}
	B, r := d.b.Slice()
	if r {
		defer opencl.Recycle(B)
	}
	opencl.AddDotProduct(slice, 1, A, B)
	return slice, true
}

func (d *dotProduct) average() []float64 {
	return qAverageUniverse(d)
}

func (d *dotProduct) Average() float64 {
	return d.average()[0]
}
