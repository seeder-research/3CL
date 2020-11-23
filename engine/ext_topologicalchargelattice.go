package engine

import (
	"math"

	"github.com/mumax/3/opencl"
	"github.com/mumax/3/data"
)

var (
	Ext_TopologicalChargeLattice        = NewScalarValue("ext_topologicalchargelattice", "", "2D topological charge", GetTopologicalChargeLattice)
	Ext_TopologicalChargeDensityLattice = NewScalarField("ext_topologicalchargedensitylattice", "1/m2",
		"2D topological charge density", SetTopologicalChargeDensityLattice)
)

func SetTopologicalChargeDensityLattice(dst *data.Slice) {
	opencl.SetTopologicalChargeLattice(dst, M.Buffer(), M.Mesh())
}

func GetTopologicalChargeLattice() float64 {
	s := ValueOf(Ext_TopologicalChargeDensityLattice)
	defer opencl.Recycle(s)
	c := Mesh().CellSize()
	N := Mesh().Size()

	return (0.25 * c[X] * c[Y] / math.Pi / float64(N[Z])) * float64(opencl.Sum(s))
}
