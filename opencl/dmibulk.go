package opencl

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Add effective field due to bulk Dzyaloshinskii-Moriya interaction to Beff.
// See dmibulk.cl
func AddDMIBulk(Beff *data.Slice, m *data.Slice, Aex_red, D_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := Beff.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	event := k_adddmibulk_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex_red), unsafe.Pointer(D_red), regions.Ptr,
		float32(cellsize[X]*1e9), float32(cellsize[Y]*1e9), float32(cellsize[Z]*1e9), N[X], N[Y], N[Z], mesh.PBC_code(), cfg,
		[](*cl.Event){Beff.GetEvent(X), Beff.GetEvent(Y), Beff.GetEvent(Z), m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})

	Beff.SetEvent(X, event)
	Beff.SetEvent(Y, event)
	Beff.SetEvent(Z, event)
	m.SetEvent(X, event)
	m.SetEvent(Y, event)
	m.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in adddmibulk: %+v \n", err)
	}
}
