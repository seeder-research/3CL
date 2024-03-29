package opencl

import (
	"fmt"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

// Set s to the toplogogical charge density s = m · (∂m/∂x ❌ ∂m/∂y)
// see topologicalcharge.cl
func SetTopologicalCharge(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	icxcy := float32(1.0 / (cellsize[X] * cellsize[Y]))

	event := k_settopologicalcharge_async(s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		icxcy, N[X], N[Y], N[Z], mesh.PBC_code(), cfg,
		[](*cl.Event){s.GetEvent(X),
			m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z)})
	s.SetEvent(X, event)
	m.SetEvent(X, event)
	m.SetEvent(Y, event)
	m.SetEvent(Z, event)

	if err := cl.WaitForEvents([]*cl.Event{event}); err != nil {
		fmt.Printf("WaitForEvents failed in settopologicalcharge: %+v \n", err)
	}
}
