package opencl

import (
	"github.com/mumax/3cl/opencl/cl"
	"github.com/mumax/3cl/opencl/mtgp32RNG"
	"github.com/mumax/3cl/timer"

	"fmt"
	"log"
	"unsafe"
)

const MTGP32_MEXP = mtgp32RNG.MTGPDC_MEXP
const MTGP32_N = mtgp32RNG.MTGPDC_N
const MTGP32_FLOOR_2P = mtgp32RNG.MTGPDC_FLOOR_2P
const MTGP32_CEIL_2P = mtgp32RNG.MTGPDC_CEIL_2P
const MTGP32_TN = mtgp32RNG.MTGPDC_TN
const MTGP32_LS = mtgp32RNG.MTGPDC_LS
const MTGP32_TS = mtgp32RNG.MTGPDC_TS

type mtgp32_params mtgp32RNG.MTGP32dc_params_array_ptr

/*
    Buffer status_buffer(context,
			 CL_MEM_READ_WRITE,
			 sizeof(uint32_t) * MTGP32_N * opt.group_num);
    buffers_t mtgp_buffers;
    mtgp_buffers.status = status_buffer;
    mtgp_buffers.rec = get_rec_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.tmp = get_tmp_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.flt = get_flt_tmp_buff(mtgp32_params_fast_11213,
					opt.group_num);
    mtgp_buffers.pos = get_pos_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.sh1 = get_sh1_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.sh2 = get_sh2_buff(mtgp32_params_fast_11213, opt.group_num);

    mtgp32 = new mtgp32_fast_t[opt.group_num];
    init_check_data(mtgp32, opt.group_num, 1234);
    initialize_by_seed(mtgp_buffers, opt.group_num, 1234); // Done
	generate_single01(mtgp_buffers, opt.group_num, opt.data_count); // Done
    free_check_data(mtgp32, opt.group_num);
*/

func NewRNGParams(group_num int) *mtgp32_params {
	var err error
	var events_list []*cl.Event
	var event *cl.Event
	tmp := mtgp32RNG.NewMTGPParams()
	tmp.GetMTGPArrays(group_num)
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

func (p *mtgp32_params) Init(group_num int, seed uint32, events []*cl.Event) {

	//	args := mtgp32_init_seed_kernel_args.argptr[:]
	event := k_mtgp32_init_seed_kernel_async ( unsafe.Pointer(p.Rec_buf), unsafe.Pointer(p.Temper_buf), unsafe.Pointer(p.Flt_temper_buf), unsafe.Pointer(p.Pos_buf),
											unsafe.Pointer(p.Sh1_buf), unsafe.Pointer(p.Sh2_buf), unsafe.Pointer(p.Status_buf), seed,
											&config{[]int{group_num * MTGP32_TN}, []int{MTGP32_TN}}, events)

	mtgp32_uniform_args.arg_param_tbl = unsafe.Pointer(p.Rec_buf)
	mtgp32_uniform_args.arg_temper_tbl = unsafe.Pointer(p.Temper_buf)
	mtgp32_uniform_args.arg_single_temper_tbl = unsafe.Pointer(p.Flt_temper_buf)
	mtgp32_uniform_args.arg_pos_tbl = unsafe.Pointer(p.Pos_buf)
	mtgp32_uniform_args.arg_sh1_tbl = unsafe.Pointer(p.Sh1_buf)
	mtgp32_uniform_args.arg_sh2_tbl = unsafe.Pointer(p.Sh2_buf)
	mtgp32_uniform_args.arg_d_status = unsafe.Pointer(p.Status_buf)

	SetKernelArgWrapper("mtgp32_uniform", 0, p.Rec_buf)
	SetKernelArgWrapper("mtgp32_uniform", 1, p.Temper_buf)
	SetKernelArgWrapper("mtgp32_uniform", 2, p.Flt_temper_buf)
	SetKernelArgWrapper("mtgp32_uniform", 3, p.Pos_buf)
	SetKernelArgWrapper("mtgp32_uniform", 4, p.Sh1_buf)
	SetKernelArgWrapper("mtgp32_uniform", 5, p.Sh2_buf)
	SetKernelArgWrapper("mtgp32_uniform", 6, p.Status_buf)

	mtgp32_normal_args.arg_param_tbl = unsafe.Pointer(p.Rec_buf)
	mtgp32_normal_args.arg_temper_tbl = unsafe.Pointer(p.Temper_buf)
	mtgp32_normal_args.arg_single_temper_tbl = unsafe.Pointer(p.Flt_temper_buf)
	mtgp32_normal_args.arg_pos_tbl = unsafe.Pointer(p.Pos_buf)
	mtgp32_normal_args.arg_sh1_tbl = unsafe.Pointer(p.Sh1_buf)
	mtgp32_normal_args.arg_sh2_tbl = unsafe.Pointer(p.Sh2_buf)
	mtgp32_normal_args.arg_d_status = unsafe.Pointer(p.Status_buf)

	SetKernelArgWrapper("mtgp32_normal", 0, p.Rec_buf)
	SetKernelArgWrapper("mtgp32_normal", 1, p.Temper_buf)
	SetKernelArgWrapper("mtgp32_normal", 2, p.Flt_temper_buf)
	SetKernelArgWrapper("mtgp32_normal", 3, p.Pos_buf)
	SetKernelArgWrapper("mtgp32_normal", 4, p.Sh1_buf)
	SetKernelArgWrapper("mtgp32_normal", 5, p.Sh2_buf)
	SetKernelArgWrapper("mtgp32_normal", 6, p.Status_buf)

	p.Ini = true
	err := cl.WaitForEvents([]*cl.Event{event})
	if err != nil {
		fmt.Printf("WaitForEvents failed in InitRNG: %+v \n", err)
	}

}

func (p *mtgp32_params) Uniform(d_data unsafe.Pointer, data_size int, group_num int, events []*cl.Event) *cl.Event {

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("mtgp32_uniform")
	}

	mtgp32_uniform_args.Lock()
	defer mtgp32_uniform_args.Unlock()

	mtgp32_uniform_args.arg_d_data = d_data
	mtgp32_uniform_args.arg_size = data_size

	item_num := MTGP32_TN * group_num
	min_size := MTGP32_LS * group_num
	if data_size%min_size != 0 {
		data_size = (data_size/min_size + 1) * min_size
	}

	SetKernelArgWrapper("mtgp32_uniform", 7, d_data)
	SetKernelArgWrapper("mtgp32_uniform", 8, data_size)

	//	args := mtgp32_uniform_args.argptr[:]
	event := LaunchKernel("mtgp32_uniform", []int{item_num}, []int{MTGP32_TN}, events)

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Stop("mtgp32_uniform")
	}

	return event
}

func (p *mtgp32_params) Normal(d_data unsafe.Pointer, data_size int, group_num int, events []*cl.Event) *cl.Event {

	if p.Ini == false {
		log.Fatalln("Generator has not been initialized!")
	}

	if Synchronous { // debug
		ClCmdQueue.Finish()
		timer.Start("mtgp32_uniform")
	}

	mtgp32_normal_args.Lock()
	defer mtgp32_normal_args.Unlock()

	mtgp32_uniform_args.arg_d_data = d_data
	mtgp32_uniform_args.arg_size = data_size

	item_num := MTGP32_TN * group_num
	min_size := MTGP32_LS * group_num
	if data_size%min_size != 0 {
		data_size = (data_size/min_size + 1) * min_size
	}

	SetKernelArgWrapper("mtgp32_normal", 7, d_data)
	SetKernelArgWrapper("mtgp32_normal", 8, data_size)

	//	args := mtgp32_uniform_args.argptr[:]
	event := LaunchKernel("mtgp32_normal", []int{item_num}, []int{MTGP32_TN}, events)

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
