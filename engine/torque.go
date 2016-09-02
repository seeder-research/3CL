package engine

import (
	"github.com/mumax/3cl/opencl"
	"github.com/mumax/3cl/data"
	"github.com/mumax/3cl/util"
)

var (
	Alpha        = NewScalarParam("alpha", "", "Landau-Lifshitz damping constant", &temp_red)
	Xi           = NewScalarParam("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol          = NewScalarParam("Pol", "", "Electrical current polarization")
	Lambda       = NewScalarParam("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime = NewScalarParam("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins  = NewScalarParam("frozenspins", "", "Defines spins that should be fixed") // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values

	FixedLayer                       = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque                           = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque                         = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque                         = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	J                                = NewExcitation("J", "A/m2", "Electrical current density")
	MaxTorque                        = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?). TODO: should not be zero
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
}

// Sets dst to the current total torque
// TODO: extensible
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	if Precess {
		opencl.LLTorque(dst, M.Buffer(), dst, Alpha.gpuLUT1(), regions.Gpu()) // overwrite dst with torque
	} else {
		opencl.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer opencl.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer opencl.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		opencl.AddZhangLiTorque(dst, M.Buffer(), jspin, Bsat.gpuLUT1(),
			Alpha.gpuLUT1(), Xi.gpuLUT1(), Pol.gpuLUT1(), regions.Gpu(), Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		opencl.AddSlonczewskiTorque(dst, M.Buffer(), jspin, fl, Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pol.gpuLUT1(), Lambda.gpuLUT1(), EpsilonPrime.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		opencl.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

// Gets
func GetMaxTorque() float64 {
	torque, recycle := Torque.Slice()
	if recycle {
		defer opencl.Recycle(torque)
	}
	return opencl.MaxVecNorm(torque)
}
