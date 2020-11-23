package opencl

import (
	"fmt"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

func SetPhi(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	event := k_setPhi_async(s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y),
		N[X], N[Y], N[Z],
		cfg, [](*cl.Event){s.GetEvent(X),
			m.GetEvent(X), m.GetEvent(Y)})
	s.SetEvent(X, event)
	m.SetEvent(X, event)
	m.SetEvent(Y, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in phi: %+v \n", err)
	}
	return
}

func SetTheta(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	event := k_setTheta_async(s.DevPtr(X), m.DevPtr(Z),
		N[X], N[Y], N[Z],
		cfg, [](*cl.Event){s.GetEvent(X),
			m.GetEvent(Z)})
	s.SetEvent(X, event)
	m.SetEvent(Z, event)
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in theta: %+v \n", err)
	}
	return
}
