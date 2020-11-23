package engine

import (
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl"
)

var (
	ext_phi   = NewScalarField("ext_phi", "rad", "Azimuthal angle", SetPhi)
	ext_theta = NewScalarField("ext_theta", "rad", "Polar angle", SetTheta)
)

func SetPhi(dst *data.Slice) {
	opencl.SetPhi(dst, M.Buffer())
}

func SetTheta(dst *data.Slice) {
	opencl.SetTheta(dst, M.Buffer())
}
