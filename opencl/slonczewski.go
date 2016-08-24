package opencl

import (
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

// Add Slonczewski ST torque to torque (Tesla).
func AddSlonczewskiTorque(torque, m, J, fixedP *data.Slice, Msat, alpha, pol, λ, ε_prime LUTPtr, regions *Bytes, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(mesh.WorldSize()[Z])

	event := k_addslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), J.DevPtr(Z),
		fixedP.DevPtr(X), fixedP.DevPtr(Y), fixedP.DevPtr(Z),
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pol),
		unsafe.Pointer(λ), unsafe.Pointer(ε_prime),
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
