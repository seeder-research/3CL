package opencl

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set s to the toplogogical charge density for lattices
// Based on the solid angle subtended by triangle associated with three spins: a,b,c
//        s = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)] / (dx dy)
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981).
// This version is best for finite-sized lattices, but does not provide a useful local density.
// See topologicalchargefinitelattice.cu
func SetTopologicalChargeFiniteLattice(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	event := k_settopologicalchargefinitelattice_async(s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N[X], N[Y], N[Z], mesh.PBC_code(),
		cfg, [](*cl.Event){s.GetEvent(X),
		        m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})

        s.SetEvent(X, event)
        m.SetEvent(X, event)
        m.SetEvent(Y, event)
        m.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in settopologicalchargefinitelattice: %+v \n", err)
	}
}
