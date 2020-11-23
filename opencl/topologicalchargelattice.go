package opencl

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set s to the toplogogical charge density for lattices
// 	  s = 2 atan[(]a . b x c /(1 + a.b + a.c + b.c) ]/ (dx dy)
// After M Boettcher et al, New J Phys 20, 103014 (2018)
// See topologicalchargelattice.cu
func SetTopologicalChargeLattice(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	icxcy := float32(1.0 / (cellsize[X] * cellsize[Y]))

	event := k_settopologicalchargelattice_async(
	        s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		icxcy, N[X], N[Y], N[Z], mesh.PBC_code(),
		cfg, [](*cl.Event){s.GetEvent(X),
		        m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})
		        
        s.SetEvent(X, event)
        m.SetEvent(X, event)
        m.SetEvent(Y, event)
        m.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in addcubicanisotropy: %+v \n", err)
	}
}
