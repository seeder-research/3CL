package opencl

import (
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
	"unsafe"
)

// Add Slonczewski ST torque to torque (Tesla).
func AddOommfSlonczewskiTorque(torque, m, J *data.Slice, fixedP *data.Slice, Msat, alpha, pfix, pfree, λfix, λfree, ε_prime LUTPtr, regions *Bytes, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(mesh.WorldSize()[Z])

	event := k_addoommfslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), J.DevPtr(Z),
		fixedP.DevPtr(X), fixedP.DevPtr(Y), fixedP.DevPtr(Z),
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pfix), unsafe.Pointer(pfree),
		unsafe.Pointer(λfix), unsafe.Pointer(λfree), unsafe.Pointer(ε_prime),
		regions.Ptr, N, cfg, [](*cl.Event){torque.GetEvent(X), torque.GetEvent(Y), torque.GetEvent(Z),
		m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z), J.GetEvent(Z),  fixedP.GetEvent(X), fixedP.GetEvent(Y),
		fixedP.GetEvent(Z)})
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
