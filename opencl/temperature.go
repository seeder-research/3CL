package opencl

import (
	"github.com/mumax/3cl/opencl/cl"

	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperature(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Msat, Temp, Alpha MSlice) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	event := k_settemperature2_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Msat.DevPtr(0), Msat.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		N, cfg,
		[]*cl.Event{Bth.GetEvent(0), noise.GetEvent(0), Msat.GetEvent(0), Temp.GetEvent(0), Alpha.GetEvent(0)})
	
	Bth.SetEvent(0, event)
	noise.SetEvent(0, event)
	Msat.SetEvent(0, event)
	Temp.SetEvent(0, event)
	Alpha.SetEvent(0, event)
}
