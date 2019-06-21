package opencl

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/mumax/3cl/opencl/RNGmtgp"
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/timer"
	"math/rand"
	"time"
)

type Prng_ interface {
	Init(uint32, []*cl.Event)
	GenerateUniform(unsafe.Pointer, int, []*cl.Event) *cl.Event
	GenerateNormal(unsafe.Pointer, int, []*cl.Event) *cl.Event
}

type Generator struct {
	Name   string
	PRNG   Prng_
	r_seed *uint32
}

const MTGP32_MEXP = RNGmtgp.MTGPDC_MEXP
const MTGP32_N = RNGmtgp.MTGPDC_N
const MTGP32_FLOOR_2P = RNGmtgp.MTGPDC_FLOOR_2P
const MTGP32_CEIL_2P = RNGmtgp.MTGPDC_CEIL_2P
const MTGP32_TN = RNGmtgp.MTGPDC_TN
const MTGP32_LS = RNGmtgp.MTGPDC_LS
const MTGP32_TS = RNGmtgp.MTGPDC_TS

type mtgp32_params RNGmtgp.MTGP32dc_params_array_ptr

func NewGenerator(name string) *Generator {
	switch name {
	case "mtgp":
		var prng_ptr Prng_
		prng_ptr = NewMTGPRNGParams()
		return &Generator{Name: "mtgp", PRNG: prng_ptr}
	case "mrg32k3a":
		fmt.Println("mrg32k3a not yet implemented")
		return nil
	default:
		fmt.Println("RNG not implemented: ", name)
		return nil
	}
}

func (g *Generator) CreatePNG() {
	switch g.Name {
	case "mtgp":
		var prng_ptr Prng_
		prng_ptr = NewMTGPRNGParams()
		g.PRNG = prng_ptr
	case "mrg32k3a":
		fmt.Println("mrg32k3a not yet implemented")
	default:
		fmt.Println("RNG not implemented: ", g.Name)
	}
}

func (g *Generator) Init(seed *uint32, events []*cl.Event) {
	if seed == nil {
		g.PRNG.Init(initRNG(), events)
	} else {
		g.PRNG.Init(*seed, events)
	}
}

func (g *Generator) Uniform(data unsafe.Pointer, d_size int, events []*cl.Event) *cl.Event {
	return g.PRNG.GenerateUniform(data, d_size, events)
}

func (g *Generator) Normal(data unsafe.Pointer, d_size int, events []*cl.Event) *cl.Event {
	return g.PRNG.GenerateNormal(data, d_size, events)
}

func NewMTGPRNGParams() *mtgp32_params {
	var err error
	var events_list []*cl.Event
	var event *cl.Event
	tmp := RNGmtgp.NewMTGPParams()
	tmp.SetGroupSize(ClCUnits)
	tmp.GetMTGPArrays()
	tmp.CreateParamBuffers(ClCtx)
	events_list, err = tmp.LoadAllParamBuffersToDevice(ClCmdQueue, nil)
	if err != nil {
		log.Fatalln("Unable to load RNG parameters to device")
	}
	event, err = tmp.LoadStatusBuffersToDevice(ClCmdQueue, nil)
	if err != nil {
		log.Fatalln("Unable to load RNG status to device")
	}
	err = cl.WaitForEvents(append(events_list, event))
	return (*mtgp32_params)(tmp)
}

func initRNG() uint32 {
	rand.Seed(time.Now().UTC().UnixNano())
	return rand.Uint32()
}

func (p *mtgp32_params) Init(seed uint32, events []*cl.Event) {

	event := k_mtgp32_init_seed_kernel_async(unsafe.Pointer(p.Rec_buf), unsafe.Pointer(p.Temper_buf), unsafe.Pointer(p.Flt_temper_buf), unsafe.Pointer(p.Pos_buf),
		unsafe.Pointer(p.Sh1_buf), unsafe.Pointer(p.Sh2_buf), unsafe.Pointer(p.Status_buf), seed,
		&config{[]int{p.GetGroupSize() * MTGP32_N}, []int{MTGP32_N}}, events)

	p.Ini = true
	err := cl.WaitForEvents([]*cl.Event{event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in InitRNG: %+v \n", err)
	}

}

func (p *mtgp32_params) GenerateUniform(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("mtgp32_uniform")
	}

	event := k_mtgp32_uniform_async(unsafe.Pointer(p.Rec_buf), unsafe.Pointer(p.Temper_buf), unsafe.Pointer(p.Flt_temper_buf), unsafe.Pointer(p.Pos_buf),
		unsafe.Pointer(p.Sh1_buf), unsafe.Pointer(p.Sh2_buf), unsafe.Pointer(p.Status_buf), d_data, data_size,
		&config{[]int{p.GetGroupSize() * MTGP32_N}, []int{MTGP32_N}}, events)

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Stop("mtgp32_uniform")
	}

	return event
}

func (p *mtgp32_params) GenerateNormal(d_data unsafe.Pointer, data_size int, events []*cl.Event) *cl.Event {

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("mtgp32_uniform")
	}

	event := k_mtgp32_normal_async(unsafe.Pointer(p.Rec_buf), unsafe.Pointer(p.Temper_buf), unsafe.Pointer(p.Flt_temper_buf), unsafe.Pointer(p.Pos_buf),
		unsafe.Pointer(p.Sh1_buf), unsafe.Pointer(p.Sh2_buf), unsafe.Pointer(p.Status_buf), d_data, data_size,
		&config{[]int{p.GetGroupSize() * MTGP32_N}, []int{MTGP32_N}}, events)

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Stop("mtgp32_normal")
	}

	return event
}

func (p *mtgp32_params) SetRecursionArray(arr []uint32) {
	p.Rec = arr
}

func (p *mtgp32_params) GetRecursionArray() []uint32 {
	return p.Rec
}

func (p *mtgp32_params) SetPositionArray(arr []int) {
	p.Pos = arr
}

func (p *mtgp32_params) GetPositionArray() []int {
	return p.Pos
}

func (p *mtgp32_params) SetSH1Array(arr []int) {
	p.Sh1 = arr
}

func (p *mtgp32_params) GetSH1Array() []int {
	return p.Sh1
}

func (p *mtgp32_params) SetSH2Array(arr []int) {
	p.Sh2 = arr
}

func (p *mtgp32_params) GetSH2Array() []int {
	return p.Sh2
}

func (p *mtgp32_params) SetStatusArray(arr []uint32) {
	p.Status = arr
}

func (p *mtgp32_params) GetStatusArray() []uint32 {
	return p.Status
}

func (p *mtgp32_params) SetGroupSize(in int) {
	p.GroupSize = in
}

func (p *mtgp32_params) GetGroupSize() int {
	return p.GroupSize
}
