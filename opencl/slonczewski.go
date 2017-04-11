package opencl

import (
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski.cl
func AddSlonczewskiTorque(torque, m *data.Slice, Msat, J, fixedP, alpha, pol, λ, ε_prime MSlice, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	flt := float32(mesh.WorldSize()[Z])

	event := k_addslonczewskitorque2_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(Z), J.Mul(Z),
		fixedP.DevPtr(X), fixedP.Mul(X),
		fixedP.DevPtr(Y), fixedP.Mul(Y),
		fixedP.DevPtr(Z), fixedP.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		pol.DevPtr(0), pol.Mul(0),
		λ.DevPtr(0), λ.Mul(0),
		ε_prime.DevPtr(0), ε_prime.Mul(0),
		unsafe.Pointer(uintptr(0)), flt,
		N, cfg,
		[](*cl.Event){torque.GetEvent(X), torque.GetEvent(Y), torque.GetEvent(Z),
        m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z),
		fixedP.GetEvent(X), fixedP.GetEvent(Y), fixedP.GetEvent(Z),
		J.GetEvent(Z)})
    torque.SetEvent(X, event)
    torque.SetEvent(Y, event)
    torque.SetEvent(Z, event)
    m.SetEvent(X, event)
    m.SetEvent(Y, event)
    m.SetEvent(Z, event)
    J.SetEvent(Z, event)
    fixedP.SetEvent(X, event)
    fixedP.SetEvent(Y, event)
    fixedP.SetEvent(Z, event)
}
