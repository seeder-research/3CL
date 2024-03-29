package opencl

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/util"
)

func SetPhi(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	sPtr_X := (unsafe.Pointer)(nil)
	mPtr_X := (unsafe.Pointer)(nil)
	mPtr_Y := (unsafe.Pointer)(nil)
	eventList := [](*cl.Event){}

	if s != nil {
		sPtr_X = s.DevPtr(X)
		eventList = append(eventList, s.GetEvent(X))
	}
	if m != nil {
		mPtr_X = m.DevPtr(X)
		eventList = append(eventList, m.GetEvent(X))
		mPtr_Y = m.DevPtr(Y)
		eventList = append(eventList, m.GetEvent(Y))
	}

	event := k_setPhi_async(sPtr_X,
		mPtr_X, mPtr_Y,
		N[X], N[Y], N[Z],
		cfg, eventList)
	if s != nil {
		s.SetEvent(X, event)
	}
	if m != nil {
		m.SetEvent(X, event)
		m.SetEvent(Y, event)
	}
	// Force synchronization
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
	sPtr_X := (unsafe.Pointer)(nil)
	mPtr_Z := (unsafe.Pointer)(nil)
	eventList := [](*cl.Event){}

	if s != nil {
		sPtr_X = s.DevPtr(X)
		eventList = append(eventList, s.GetEvent(X))
	}
	if m != nil {
		mPtr_Z = m.DevPtr(Z)
		eventList = append(eventList, m.GetEvent(Z))
	}

	event := k_setTheta_async(sPtr_X, mPtr_Z,
		N[X], N[Y], N[Z],
		cfg, eventList)
	if s != nil {
		s.SetEvent(X, event)
	}
	if m != nil {
		m.SetEvent(Z, event)
	}
	// Force synchronization
	err := cl.WaitForEvents([](*cl.Event){event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in theta: %+v \n", err)
	}
	return
}
