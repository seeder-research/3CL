package opencl

import (
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/data"
)

// Add Zhang-Li ST torque (Tesla) to torque.
func AddZhangLiTorque(torque, m, J *data.Slice, bsat, alpha, xi, pol LUTPtr, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	event := k_addzhanglitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		J.DevPtr(X), J.DevPtr(Y), J.DevPtr(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		unsafe.Pointer(bsat), unsafe.Pointer(alpha), unsafe.Pointer(xi), unsafe.Pointer(pol),
		regions.Ptr, N[X], N[Y], N[Z], mesh.PBC_code(), cfg, [](*cl.Event){torque.GetEvent(X),
		torque.GetEvent(Y), torque.GetEvent(Z), m.GetEvent(X), m.GetEvent(Y), m.GetEvent(Z),
		J.GetEvent(X), J.GetEvent(Y), J.GetEvent(Z)})
        torque.SetEvent(X, event)
        torque.SetEvent(Y, event)
        torque.SetEvent(Z, event)
        m.SetEvent(X, event)
        m.SetEvent(Y, event)
        m.SetEvent(Z, event)
        J.SetEvent(X, event)
        J.SetEvent(Y, event)
        J.SetEvent(Z, event)
}
