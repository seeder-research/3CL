package opencl

import (
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

// Adds cubic anisotropy field to Beff.
// see cubicanisotropy.cu
func AddCubicAnisotropy(Beff, m *data.Slice, k1_red, k2_red, k3_red LUTPtr, c1, c2 LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addcubicanisotropy_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(k1_red), unsafe.Pointer(k2_red), unsafe.Pointer(k3_red),
		c1[X], c1[Y], c1[Z],
		c2[X], c2[Y], c2[Z],
		regions.Ptr, N, cfg, [](*cl.Event){Beff.GetEvent(X), 
		Beff.GetEvent(Y), Beff.GetEvent(Z), m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})
}

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// see uniaxialanisotropy.cu
func AddUniaxialAnisotropy(Beff, m *data.Slice, k1_red, k2_red LUTPtr, u LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(k1_red), unsafe.Pointer(k2_red),
		u[X], u[Y], u[Z],
		regions.Ptr, N, cfg, [](*cl.Event){Beff.GetEvent(X),
                Beff.GetEvent(Y), Beff.GetEvent(Z), m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})
}
