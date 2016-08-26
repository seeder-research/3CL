/*
Package clFFT provides a binding to the clFFT api.

Resource life-cycle management:

For any CL object that gets created (buffer, queue, kernel, etc..) you should
call object.Release() when finished with it to free the CL resources. This
explicitely calls the related clfftXXXRelease method for the type. However,
as a fallback there is a finalizer set for every resource item that takes
care of it (eventually) if Release isn't called. In this way you can have
better control over the life cycle of resources while having a fall back
to avoid leaks. This is similar to how file handles and such are handled
in the Go standard packages.
*/
package clFFT

/*
#include <stdlib.h>
#include <clFFT.h>

extern void go_bakeplan_notify(clfftPlanHandle plHandle, void *user_data);
static inline void CL_CALLBACK c_bakeplan_notify(clfftPlanHandle plHandle, void *user_data){
	go_bakeplan_notify(plHandle, user_data);
}

static clfftStatus CLFFTBakePlan(clfftPlanHandle	plHandle,
				 cl_uint		numQueues,
				 cl_command_queue*	commQueueFFT,
				 void* user_data){
	return clfftBakePlan(plHandle, numQueues, commQueueFFT, c_bakeplan_notify, user_data);
}

static cl_context FromVoidToClContext(void *memptr){
	cl_context * val;
	val = memptr;
	return *val;
}

static cl_command_queue FromVoidToClCommandQueue(void *memptr){
        cl_command_queue * val;
        val = memptr;
        return *val;
}

static cl_event FromVoidToClEvent(void *memptr){
        cl_event * val;
        val = memptr;
        return *val;
}

static cl_mem FromVoidToClMem(void *memptr){
        cl_mem * val;
        val = memptr;
        return *val;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/mumax/3cl/opencl/cl"
)

//////////////// Basic Errors ////////////////

var ErrUnsupported = errors.New("clFFT: unsupported")

var (
	ErrUnknown = errors.New("clFFT: unknown error") // Generally an unexpected result from a clFFT function (e.g. CL_SUCCESS but null pointer)
)

type ErrOther int

func (e ErrOther) Error() string {
	return fmt.Sprintf("clFFT: error %d", int(e))
}

var (
	ErrDeviceNotFound               = errors.New("clFFT: Device Not Found")
	ErrDeviceNotAvailable           = errors.New("clFFT: Device Not Available")
	ErrCompilerNotAvailable         = errors.New("clFFT: Compiler Not Available")
	ErrMemObjectAllocationFailure   = errors.New("clFFT: Mem Object Allocation Failure")
	ErrOutOfResources               = errors.New("clFFT: Out Of Resources")
	ErrOutOfHostMemory              = errors.New("clFFT: Out Of Host Memory")
	ErrProfilingInfoNotAvailable    = errors.New("clFFT: Profiling Info Not Available")
	ErrMemCopyOverlap               = errors.New("clFFT: Mem Copy Overlap")
	ErrImageFormatMismatch          = errors.New("clFFT: Image Format Mismatch")
	ErrImageFormatNotSupported      = errors.New("clFFT: Image Format Not Supported")
	ErrBuildProgramFailure          = errors.New("clFFT: Build Program Failure")
	ErrMapFailure                   = errors.New("clFFT: Map Failure")
	ErrInvalidValue                 = errors.New("clFFT: Invalid Value")
	ErrInvalidDeviceType            = errors.New("clFFT: Invalid Device Type")
	ErrInvalidPlatform              = errors.New("clFFT: Invalid Platform")
	ErrInvalidDevice                = errors.New("clFFT: Invalid Device")
	ErrInvalidContext               = errors.New("clFFT: Invalid Context")
	ErrInvalidQueueProperties       = errors.New("clFFT: Invalid Queue Properties")
	ErrInvalidCommandQueue          = errors.New("clFFT: Invalid Command Queue")
	ErrInvalidHostPtr               = errors.New("clFFT: Invalid Host Ptr")
	ErrInvalidMemObject             = errors.New("clFFT: Invalid Mem Object")
	ErrInvalidImageFormatDescriptor = errors.New("clFFT: Invalid Image Format Descriptor")
	ErrInvalidImageSize             = errors.New("clFFT: Invalid Image Size")
	ErrInvalidSampler               = errors.New("clFFT: Invalid Sampler")
	ErrInvalidBinary                = errors.New("clFFT: Invalid Binary")
	ErrInvalidBuildOptions          = errors.New("clFFT: Invalid Build Options")
	ErrInvalidProgram               = errors.New("clFFT: Invalid Program")
	ErrInvalidProgramExecutable     = errors.New("clFFT: Invalid Program Executable")
	ErrInvalidKernelName            = errors.New("clFFT: Invalid Kernel Name")
	ErrInvalidKernelDefinition      = errors.New("clFFT: Invalid Kernel Definition")
	ErrInvalidKernel                = errors.New("clFFT: Invalid Kernel")
	ErrInvalidArgIndex              = errors.New("clFFT: Invalid Arg Index")
	ErrInvalidArgValue              = errors.New("clFFT: Invalid Arg Value")
	ErrInvalidArgSize               = errors.New("clFFT: Invalid Arg Size")
	ErrInvalidKernelArgs            = errors.New("clFFT: Invalid Kernel Args")
	ErrInvalidWorkDimension         = errors.New("clFFT: Invalid Work Dimension")
	ErrInvalidWorkGroupSize         = errors.New("clFFT: Invalid Work Group Size")
	ErrInvalidWorkItemSize          = errors.New("clFFT: Invalid Work Item Size")
	ErrInvalidGlobalOffset          = errors.New("clFFT: Invalid Global Offset")
	ErrInvalidEventWaitList         = errors.New("clFFT: Invalid Event Wait List")
	ErrInvalidEvent                 = errors.New("clFFT: Invalid Event")
	ErrInvalidOperation             = errors.New("clFFT: Invalid Operation")
	ErrInvalidBufferSize            = errors.New("clFFT: Invalid Buffer Size")
	ErrInvalidGlobalWorkSize        = errors.New("clFFT: Invalid Global Work Size")
	ErrInvalidProperty              = errors.New("clFFT: Invalid Property")
	ErrInvalidImageDescriptor       = errors.New("clFFT: Invalid Image Descriptor")
	ErrInvalidCompilerOptions       = errors.New("clFFT: Invalid Compiler Options")
	ErrInvalidLinkerOptions         = errors.New("clFFT: Invalid Linker Options")
	ErrInvalidDevicePartitionCount  = errors.New("clFFT: Invalid Device Partition Count")
	ErrInvalidGLObject              = errors.New("clFFT: Invalid GL Object")
	ErrInvalidMipLevel              = errors.New("clFFT: Invalid Mip Level")
	ErrCLFFTInvalidPlan             = errors.New("clFFT: Invalid Plan")

	ErrCLFFTBugCheck                = errors.New("clFFT: Bug Check")
	ErrCLFFTNotImplemented          = errors.New("clFFT: Not Implemented")
	ErrCLFFTTransposeNotImplemented = errors.New("clFFT: Transpose Not Implemented")
	ErrCLFFTFileNotFound            = errors.New("clFFT: File Not Found")
	ErrCLFFTFileCreateFailure       = errors.New("clFFT: File Creation Failure")
	ErrCLFFTVersionMismatch         = errors.New("clFFT: Version Mismatch")
	ErrCLFFTNoDouble                = errors.New("clFFT: Device Does Not Support Double")
	ErrCLFFTDeviceMismatch          = errors.New("clFFT: Device Mismatch")
	ErrCLFFTEndStatus               = errors.New("clFFT: End Status")
)

var errorMap = map[C.clfftStatus]error{
	C.CLFFT_SUCCESS:                         nil,
	C.CLFFT_INVALID_GLOBAL_WORK_SIZE:        ErrInvalidGlobalWorkSize,
	C.CLFFT_INVALID_MIP_LEVEL:               ErrInvalidMipLevel,
	C.CLFFT_INVALID_BUFFER_SIZE:             ErrInvalidBufferSize,
	C.CLFFT_INVALID_GL_OBJECT:               ErrInvalidGLObject,
	C.CLFFT_INVALID_OPERATION:               ErrInvalidOperation,
	C.CLFFT_INVALID_EVENT:                   ErrInvalidEvent,
	C.CLFFT_INVALID_EVENT_WAIT_LIST:         ErrInvalidEventWaitList,
	C.CLFFT_INVALID_GLOBAL_OFFSET:           ErrInvalidGlobalOffset,
	C.CLFFT_INVALID_WORK_ITEM_SIZE:          ErrInvalidWorkItemSize,
	C.CLFFT_INVALID_WORK_GROUP_SIZE:         ErrInvalidWorkGroupSize,
	C.CLFFT_INVALID_WORK_DIMENSION:          ErrInvalidWorkDimension,
	C.CLFFT_INVALID_KERNEL_ARGS:             ErrInvalidKernelArgs,
	C.CLFFT_INVALID_ARG_SIZE:                ErrInvalidArgSize,
	C.CLFFT_INVALID_ARG_VALUE:               ErrInvalidArgValue,
	C.CLFFT_INVALID_ARG_INDEX:               ErrInvalidArgIndex,
	C.CLFFT_INVALID_KERNEL:                  ErrInvalidKernel,
	C.CLFFT_INVALID_KERNEL_DEFINITION:       ErrInvalidKernelDefinition,
	C.CLFFT_INVALID_KERNEL_NAME:             ErrInvalidKernelName,
	C.CLFFT_INVALID_PROGRAM_EXECUTABLE:      ErrInvalidProgramExecutable,
	C.CLFFT_INVALID_PROGRAM:                 ErrInvalidProgram,
	C.CLFFT_INVALID_BUILD_OPTIONS:           ErrInvalidBuildOptions,
	C.CLFFT_INVALID_BINARY:                  ErrInvalidBinary,
	C.CLFFT_INVALID_SAMPLER:                 ErrInvalidSampler,
	C.CLFFT_INVALID_IMAGE_SIZE:              ErrInvalidImageSize,
	C.CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR: ErrInvalidImageFormatDescriptor,
	C.CLFFT_INVALID_MEM_OBJECT:              ErrInvalidMemObject,
	C.CLFFT_INVALID_HOST_PTR:                ErrInvalidHostPtr,
	C.CLFFT_INVALID_COMMAND_QUEUE:           ErrInvalidCommandQueue,
	C.CLFFT_INVALID_QUEUE_PROPERTIES:        ErrInvalidQueueProperties,
	C.CLFFT_INVALID_CONTEXT:                 ErrInvalidContext,
	C.CLFFT_INVALID_DEVICE:                  ErrInvalidDevice,
	C.CLFFT_INVALID_PLATFORM:                ErrInvalidPlatform,
	C.CLFFT_INVALID_DEVICE_TYPE:             ErrInvalidDeviceType,
	C.CLFFT_INVALID_VALUE:                   ErrInvalidValue,
	C.CLFFT_MAP_FAILURE:                     ErrMapFailure,
	C.CLFFT_BUILD_PROGRAM_FAILURE:           ErrBuildProgramFailure,
	C.CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:      ErrImageFormatNotSupported,
	C.CLFFT_IMAGE_FORMAT_MISMATCH:           ErrImageFormatMismatch,
	C.CLFFT_MEM_COPY_OVERLAP:                ErrMemCopyOverlap,
	C.CLFFT_PROFILING_INFO_NOT_AVAILABLE:    ErrProfilingInfoNotAvailable,
	C.CLFFT_OUT_OF_HOST_MEMORY:              ErrOutOfHostMemory,
	C.CLFFT_OUT_OF_RESOURCES:                ErrOutOfResources,
	C.CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:   ErrMemObjectAllocationFailure,
	C.CLFFT_COMPILER_NOT_AVAILABLE:          ErrCompilerNotAvailable,
	C.CLFFT_DEVICE_NOT_AVAILABLE:            ErrDeviceNotAvailable,
	C.CLFFT_DEVICE_NOT_FOUND:                ErrDeviceNotFound,
	C.CLFFT_BUGCHECK:                        ErrCLFFTBugCheck,
	C.CLFFT_NOTIMPLEMENTED:                  ErrCLFFTNotImplemented,
	C.CLFFT_TRANSPOSED_NOTIMPLEMENTED:       ErrCLFFTTransposeNotImplemented,
	C.CLFFT_FILE_NOT_FOUND:                  ErrCLFFTFileNotFound,
	C.CLFFT_FILE_CREATE_FAILURE:             ErrCLFFTFileCreateFailure,
	C.CLFFT_VERSION_MISMATCH:                ErrCLFFTVersionMismatch,
	C.CLFFT_INVALID_PLAN:                    ErrCLFFTInvalidPlan,
	C.CLFFT_DEVICE_NO_DOUBLE:                ErrCLFFTNoDouble,
	C.CLFFT_DEVICE_MISMATCH:                 ErrCLFFTDeviceMismatch,
	C.CLFFT_ENDSTATUS:                       ErrCLFFTEndStatus,
}

func toError(code C.clfftStatus) error {
	if err, ok := errorMap[code]; ok {
		return err
	}
	return ErrOther(code)
}

//////////////// Basic Types ////////////////

type ClFFTStatus int

type ClFFTDim int

const (
	CLFFTDim1D ClFFTDim = C.CLFFT_1D
	CLFFTDim2D ClFFTDim = C.CLFFT_2D
	CLFFTDim3D ClFFTDim = C.CLFFT_3D
)

type ClFFTLayout int

const (
	CLFFTLayoutComplexInterleaved   ClFFTLayout = C.CLFFT_COMPLEX_INTERLEAVED
	CLFFTLayoutComplexPlanar        ClFFTLayout = C.CLFFT_COMPLEX_PLANAR
	CLFFTLayoutHermitianInterleaved ClFFTLayout = C.CLFFT_HERMITIAN_INTERLEAVED
	CLFFTLayoutHermitianPlanar      ClFFTLayout = C.CLFFT_HERMITIAN_PLANAR
	CLFFTLayoutReal                 ClFFTLayout = C.CLFFT_REAL
)

var layoutMap = map[ClFFTLayout]C.clfftLayout{
	CLFFTLayoutComplexInterleaved:   C.CLFFT_COMPLEX_INTERLEAVED,
	CLFFTLayoutComplexPlanar:        C.CLFFT_COMPLEX_PLANAR,
	CLFFTLayoutHermitianInterleaved: C.CLFFT_HERMITIAN_INTERLEAVED,
	CLFFTLayoutHermitianPlanar:      C.CLFFT_HERMITIAN_PLANAR,
	CLFFTLayoutReal:                 C.CLFFT_REAL,
}

func (layout *ClFFTLayout) toCl() C.clfftLayout {
	return layoutMap[*layout]
}

type ClFFTPrecision int

const (
	CLFFTPrecisionSingle     ClFFTPrecision = C.CLFFT_SINGLE
	CLFFTPrecisionDouble     ClFFTPrecision = C.CLFFT_DOUBLE
	CLFFTPrecisionFastSingle ClFFTPrecision = C.CLFFT_SINGLE_FAST
	CLFFTPrecisionFastDouble ClFFTPrecision = C.CLFFT_DOUBLE_FAST
)

type ClFFTDirection int

const (
	ClFFTDirectionForward  ClFFTDirection = C.CLFFT_FORWARD
	ClFFTDirectionBackward ClFFTDirection = C.CLFFT_BACKWARD
	ClFFTDirectionMinus    ClFFTDirection = C.CLFFT_MINUS
	ClFFTDirectionPlus     ClFFTDirection = C.CLFFT_PLUS
)

type ClFFTResultLocation int

const (
	ClFFTResultLocationInplace    ClFFTResultLocation = C.CLFFT_INPLACE
	ClFFTResultLocationOutOfPlace ClFFTResultLocation = C.CLFFT_OUTOFPLACE
)

type ClFFTResultTransposed int

const (
	ClFFTResultNoTranspose ClFFTResultTransposed = C.CLFFT_NOTRANSPOSE
	ClFFTResultTranspose   ClFFTResultTransposed = C.CLFFT_TRANSPOSED
)

type ClFFTCallbackType int

const (
	ClFFTCallbackTypePre  ClFFTCallbackType = C.PRECALLBACK
	ClFFTCallbackTypePost ClFFTCallbackType = C.POSTCALLBACK
)

type ClFFTPlan struct {
	clFFTHandle C.clfftPlanHandle
}

// clfftSetupData is a data structure consisting of cl_uint variables major, minor, and patch, and cl_ulong variable debugFlags
// which indicates the API version and behavior
var ClFFTSetupData C.clfftSetupData

type ClFFTDebugFlag int

const (
	CLFFTDumpPrograms ClFFTDebugFlag = C.CLFFT_DUMP_PROGRAMS
)

//////////////// Abstract Types ////////////////
type ArrayDistances struct {
	inputs  int
	outputs int
}

type ArrayLayouts struct {
	inputs  ClFFTLayout
	outputs ClFFTLayout
}

//////////////// Supporting Types ////////////////
type CLFFT_BakePlan_notify func(C.clfftPlanHandle, unsafe.Pointer)
var bakeplan_notify map[unsafe.Pointer]CLFFT_BakePlan_notify

//////////////// Basic Functions ////////////////
func init() {
	if err := C.clfftInitSetupData(&ClFFTSetupData); err != C.CLFFT_SUCCESS {
		panic("Shouldn't fail!")
		fmt.Printf("failed to initialize clFFT: %+v \n", toError(err))
	}
	bakeplan_notify = make(map[unsafe.Pointer]CLFFT_BakePlan_notify)
}

//export go_bakeplan_notify
func go_bakeplan_notify(plHandle C.clfftPlanHandle, user_data unsafe.Pointer) {
        var c_user_data []unsafe.Pointer
        c_user_data = *(*[]unsafe.Pointer)(user_data)
        bakeplan_notify[c_user_data[1]](plHandle, c_user_data[0])
}

func SetupCLFFT() error {
	return toError(C.clfftSetup(&ClFFTSetupData))
}

func TeardownCLFFT() error {
	return toError(C.clfftTeardown())
}

func GetCLFFTVersion() (int, int, int) {
	var major, minor, patch C.cl_uint
	if err := C.clfftGetVersion(&major, &minor, &patch); err != C.CLFFT_SUCCESS {
		fmt.Printf("failed to get clFFT version information: %+v \n", toError(err))
		return -1, -1, -1
	}
	return int(major), int(minor), int(patch)
}

func NewCLFFTPlan(ctx *cl.Context, dim ClFFTDim, dLengths []int) (*ClFFTPlan, error) {
	var dimInt int
	switch dim {
	default:
		dimInt = 1
	case CLFFTDim1D:
		dimInt = 1
	case CLFFTDim2D:
		dimInt = 2
	case CLFFTDim3D:
		dimInt = 3
	}
	if len(dLengths) != dimInt {
		return nil, ErrInvalidValue
	}
	cLengths := make([]C.size_t, dimInt)
	for idx, val := range dLengths {
		if val < 1 {
			fmt.Printf("invalid length defined for dimension %d: ( %d ) \n", idx, val)
		}
		cLengths[idx] = C.size_t(val)
	}
	var outPlanHandle C.clfftPlanHandle
	if err := C.clfftCreateDefaultPlan(&outPlanHandle, (C.FromVoidToClContext(ctx.ToCl())), C.clfftDim(dim), &cLengths[0]); err != C.CLFFT_SUCCESS {
		panic("failed to create default clfft plan!")
		return nil, toError(err)
	}
	return &ClFFTPlan{outPlanHandle}, nil
}

func NewArrayLayout() (*ArrayLayouts) {
	return &ArrayLayouts{inputs: CLFFTLayoutComplexInterleaved, outputs: CLFFTLayoutComplexInterleaved}
}

//////////////// Abstract Functions ////////////////
func (FFTplan *ClFFTPlan) CopyPlan(ctx *cl.Context) (*ClFFTPlan, error) {
	var outPlanHandle C.clfftPlanHandle
	if err := C.clfftCopyPlan(&outPlanHandle, (C.FromVoidToClContext(ctx.ToCl())), FFTplan.clFFTHandle); err != C.CLFFT_SUCCESS {
		panic("failed to copy clfft plan!")
		return nil, toError(err)
	}
	return &ClFFTPlan{outPlanHandle}, nil
}

func (FFTplan *ClFFTPlan) BakePlanSimple(CommQueues []*cl.CommandQueue) error {
        QueueList := make([]C.cl_command_queue, len(CommQueues))
        for idx, id := range CommQueues {
                tmp := id.GetQueueID()
                QueueList[idx] = C.FromVoidToClCommandQueue(unsafe.Pointer(&tmp))
        }
	return toError(C.clfftBakePlan(FFTplan.clFFTHandle, C.cl_uint(len(QueueList)), &QueueList[0], nil, nil))
}

func (FFTplan *ClFFTPlan) BakePlanUnsafe(CommQueues []*cl.CommandQueue, user_data unsafe.Pointer) error {
        QueueList := make([]C.cl_command_queue, len(CommQueues))
        for idx, id := range CommQueues {
                tmp := id.GetQueueID()
                QueueList[idx] = C.FromVoidToClCommandQueue(unsafe.Pointer(&tmp))
        }
        return toError(C.CLFFTBakePlan(FFTplan.clFFTHandle, C.cl_uint(len(QueueList)), &QueueList[0], user_data))
}

func (FFTplan *ClFFTPlan) Destroy() error {
	return toError(C.clfftDestroyPlan(&FFTplan.clFFTHandle))
}

func (FFTplan *ClFFTPlan) GetContext() (*cl.Context, error) {
	var outVal C.cl_context
	err := C.clfftGetPlanContext(FFTplan.clFFTHandle, &outVal)
	return cl.NewDevlessContext(unsafe.Pointer(&outVal)), toError(err)
}

func (FFTplan *ClFFTPlan) GetPrecision() (ClFFTPrecision, error) {
	var paramVal C.clfftPrecision
	defer C.free(paramVal)
	if err := C.clfftGetPlanPrecision(FFTplan.clFFTHandle, &paramVal); err != C.CLFFT_SUCCESS {
		fmt.Printf("failed to get precision of clfft plan \n")
		return -1, toError(err)
	}
	switch paramVal {
	default:
		return -1, nil
	case C.CLFFT_SINGLE:
		return CLFFTPrecisionSingle, nil
	case C.CLFFT_DOUBLE:
		return CLFFTPrecisionDouble, nil
	case C.CLFFT_SINGLE_FAST:
		return CLFFTPrecisionFastSingle, nil
	case C.CLFFT_DOUBLE_FAST:
		return CLFFTPrecisionFastDouble, nil
	}
}

func (FFTplan *ClFFTPlan) SetSinglePrecision() error {
	return toError(C.clfftSetPlanPrecision(FFTplan.clFFTHandle, C.CLFFT_SINGLE))
}

func (FFTplan *ClFFTPlan) SetDoublePrecision() error {
	return toError(C.clfftSetPlanPrecision(FFTplan.clFFTHandle, C.CLFFT_DOUBLE))
}

func (FFTplan *ClFFTPlan) SetFastSinglePrecision() error {
	return toError(C.clfftSetPlanPrecision(FFTplan.clFFTHandle, C.CLFFT_SINGLE_FAST))
}

func (FFTplan *ClFFTPlan) SetFastDoublePrecision() error {
	return toError(C.clfftSetPlanPrecision(FFTplan.clFFTHandle, C.CLFFT_DOUBLE_FAST))
}

func (FFTplan *ClFFTPlan) GetScale(direction ClFFTDirection) (float32, error) {
	var scaleDir C.clfftDirection
	var outVal C.cl_float
	defer C.free(scaleDir)
	switch direction {
	default:
		scaleDir = C.CLFFT_FORWARD
	case ClFFTDirectionForward:
		scaleDir = C.CLFFT_FORWARD
	case ClFFTDirectionBackward:
		scaleDir = C.CLFFT_BACKWARD
	}
	err := C.clfftGetPlanScale(FFTplan.clFFTHandle, scaleDir, &outVal)
	return float32(outVal), toError(err)
}

func (FFTplan *ClFFTPlan) SetScale(direction ClFFTDirection, scale float32) error {
	var scaleDir C.clfftDirection
	defer C.free(scaleDir)
	switch direction {
	default:
		scaleDir = C.CLFFT_FORWARD
	case ClFFTDirectionForward:
		scaleDir = C.CLFFT_FORWARD
	case ClFFTDirectionBackward:
		scaleDir = C.CLFFT_BACKWARD
	}
	return toError(C.clfftSetPlanScale(FFTplan.clFFTHandle, scaleDir, C.cl_float(scale)))
}

func (FFTplan *ClFFTPlan) GetBatchSize() (int, error) {
	var outVal C.size_t
	err := C.clfftGetPlanBatchSize(FFTplan.clFFTHandle, &outVal)
	return int(outVal), toError(err)
}

func (FFTplan *ClFFTPlan) SetBatchSize(batchSize int) error {
	return toError(C.clfftSetPlanBatchSize(FFTplan.clFFTHandle, C.size_t(batchSize)))
}

func (FFTplan *ClFFTPlan) GetDim() (ClFFTDim, error) {
	var outVal C.clfftDim
	defer C.free(outVal)
	err := C.clfftGetPlanDim(FFTplan.clFFTHandle, &outVal, nil)
	switch outVal {
	default:
		return -1, toError(err)
	case C.CLFFT_1D:
		return CLFFTDim1D, toError(err)
	case C.CLFFT_2D:
		return CLFFTDim2D, toError(err)
	case C.CLFFT_3D:
		return CLFFTDim3D, toError(err)
	}
}

func (FFTplan *ClFFTPlan) SetDim(inVal ClFFTDim) error {
	switch inVal {
	default:
		return toError(C.CLFFT_INVALID_VALUE)
	case CLFFTDim1D:
		return toError(C.clfftSetPlanDim(FFTplan.clFFTHandle, C.CLFFT_1D))
	case CLFFTDim2D:
		return toError(C.clfftSetPlanDim(FFTplan.clFFTHandle, C.CLFFT_2D))
	case CLFFTDim3D:
		return toError(C.clfftSetPlanDim(FFTplan.clFFTHandle, C.CLFFT_3D))
	}
}

func (FFTplan *ClFFTPlan) GetLength() ([]int, error) {
	var outVal C.clfftDim
	var outArrLength C.cl_uint
	var err C.clfftStatus
	defer C.free(outVal)
	defer C.free(outArrLength)
	defer C.free(err)
	if err = C.clfftGetPlanDim(FFTplan.clFFTHandle, &outVal, &outArrLength); err != C.CLFFT_SUCCESS {
		return []int{}, toError(err)
	}
	outList := make([]C.size_t, int(outArrLength))
	err = C.clfftGetPlanLength(FFTplan.clFFTHandle, outVal, &outList[0])
	if err = C.clfftGetPlanLength(FFTplan.clFFTHandle, outVal, &outList[0]); err != C.CLFFT_SUCCESS {
		return []int{}, toError(err)
	}
	goList := make([]int, int(outArrLength))
	for idx, val := range outList {
		goList[idx] = int(val)
	}
	return goList, nil
}

func (FFTplan *ClFFTPlan) Set1DLength(inVal int) error {
	val := C.size_t(inVal)
	defer C.free(val)
	return toError(C.clfftSetPlanLength(FFTplan.clFFTHandle, C.CLFFT_1D, &val))
}

func (FFTplan *ClFFTPlan) Set2DLength(inVal []int) error {
	if len(inVal) != 2 {
		return ErrInvalidValue
	}
	val := make([]C.size_t, 2)
	val[0], val[1] = C.size_t(inVal[0]), C.size_t(inVal[1])
	defer C.free(val)
	return toError(C.clfftSetPlanLength(FFTplan.clFFTHandle, C.CLFFT_2D, &val[0]))
}

func (FFTplan *ClFFTPlan) Set3DLength(inVal []int) error {
	if len(inVal) != 3 {
		return ErrInvalidValue
	}
	val := make([]C.size_t, 3)
	val[0], val[1], val[2] = C.size_t(inVal[0]), C.size_t(inVal[1]), C.size_t(inVal[2])
	defer C.free(val)
	return toError(C.clfftSetPlanLength(FFTplan.clFFTHandle, C.CLFFT_3D, &val[0]))
}

func (FFTplan *ClFFTPlan) GetInStride() ([]int, error) {
	planDim, err := FFTplan.GetDim()
	if err != nil {
		return []int{}, err
	}
	switch planDim {
	default:
		return []int{}, err
	case CLFFTDim1D:
		return []int{1}, nil
	case CLFFTDim2D:
		returnVal := make([]C.size_t, 2)
		err = toError(C.clfftGetPlanInStride(FFTplan.clFFTHandle, C.CLFFT_2D, &returnVal[0]))
		if err != nil {
			return []int{}, err
		}
		return []int{int(returnVal[0]), int(returnVal[1])}, nil
	case CLFFTDim3D:
		returnVal := make([]C.size_t, 3)
		err = toError(C.clfftGetPlanInStride(FFTplan.clFFTHandle, C.CLFFT_3D, &returnVal[0]))
		if err != nil {
			return []int{}, err
		}
		return []int{int(returnVal[0]), int(returnVal[1]), int(returnVal[2])}, nil
	}
}

func (FFTplan *ClFFTPlan) SetInStride(strides []int) error {
	strideLength := len(strides)
	if (strideLength < 2) || (strideLength > 3) {
		return ErrInvalidValue
	}
	if strideLength == 2 {
		inVal := make([]C.size_t, 2)
		inVal[0], inVal[1] = C.size_t(strides[0]), C.size_t(strides[1])
		return toError(C.clfftSetPlanInStride(FFTplan.clFFTHandle, C.CLFFT_2D, &inVal[0]))
	}
	inVal := make([]C.size_t, 3)
	inVal[0], inVal[1], inVal[2] = C.size_t(strides[0]), C.size_t(strides[1]), C.size_t(strides[2])
	return toError(C.clfftSetPlanInStride(FFTplan.clFFTHandle, C.CLFFT_3D, &inVal[0]))
}

func (FFTplan *ClFFTPlan) GetOutStride() ([]int, error) {
	planDim, err := FFTplan.GetDim()
	if err != nil {
		return []int{}, err
	}
	switch planDim {
	default:
		return []int{}, err
	case CLFFTDim1D:
		return []int{1}, nil
	case CLFFTDim2D:
		returnVal := make([]C.size_t, 2)
		err = toError(C.clfftGetPlanOutStride(FFTplan.clFFTHandle, C.CLFFT_2D, &returnVal[0]))
		if err != nil {
			return []int{}, err
		}
		return []int{int(returnVal[0]), int(returnVal[1])}, nil
	case CLFFTDim3D:
		returnVal := make([]C.size_t, 3)
		err = toError(C.clfftGetPlanOutStride(FFTplan.clFFTHandle, C.CLFFT_3D, &returnVal[0]))
		if err != nil {
			return []int{}, err
		}
		return []int{int(returnVal[0]), int(returnVal[1]), int(returnVal[2])}, nil
	}
}

func (FFTplan *ClFFTPlan) SetOutStride(strides []int) error {
	strideLength := len(strides)
	if (strideLength < 2) || (strideLength > 3) {
		return ErrInvalidValue
	}
	if strideLength == 2 {
		inVal := make([]C.size_t, 2)
		inVal[0], inVal[1] = C.size_t(strides[0]), C.size_t(strides[1])
		return toError(C.clfftSetPlanOutStride(FFTplan.clFFTHandle, C.CLFFT_2D, &inVal[0]))
	}
	inVal := make([]C.size_t, 3)
	inVal[0], inVal[1], inVal[2] = C.size_t(strides[0]), C.size_t(strides[1]), C.size_t(strides[2])
	return toError(C.clfftSetPlanOutStride(FFTplan.clFFTHandle, C.CLFFT_3D, &inVal[0]))
}

func (FFTplan *ClFFTPlan) GetDistances() (*ArrayDistances, error) {
	var inDist, outDist C.size_t
	err := toError(C.clfftGetPlanDistance(FFTplan.clFFTHandle, &inDist, &outDist))
	return &ArrayDistances{inputs: int(inDist), outputs: int(outDist)}, err
}

func (FFTplan *ClFFTPlan) SetDistances(distances *ArrayDistances) error {
	return toError(C.clfftSetPlanDistance(FFTplan.clFFTHandle, C.size_t(distances.inputs), C.size_t(distances.outputs)))
}

func (FFTplan *ClFFTPlan) GetLayouts() (*ArrayLayouts, error) {
        var inVal, outVal C.clfftLayout
        err := toError(C.clfftGetLayout(FFTplan.clFFTHandle, &inVal, &outVal))
        return &ArrayLayouts{inputs: ClFFTLayout(inVal), outputs: ClFFTLayout(outVal)}, err
}

func (FFTplan *ClFFTPlan) SetLayouts(layouts *ArrayLayouts) error {
        return toError(C.clfftSetLayout(FFTplan.clFFTHandle, layouts.inputs.toCl(), layouts.outputs.toCl()))
}

// GetResultPlacement returns true if the transform result is stored back into the input buffer (inplace transform)
func (FFTplan *ClFFTPlan) GetResultPlacement() (bool, error) {
	var val C.clfftResultLocation
	err := toError(C.clfftGetResultLocation(FFTplan.clFFTHandle, &val))
	if err != nil {
		return false, err
	}
	switch val {
	default:
		return false, err
	case C.CLFFT_INPLACE:
		return true, nil
	case C.CLFFT_OUTOFPLACE:
		return false, nil
	}
}

func (FFTplan *ClFFTPlan) SetResultInplace() error {
	return toError(C.clfftSetResultLocation(FFTplan.clFFTHandle, C.CLFFT_INPLACE))
}

func (FFTplan *ClFFTPlan) SetResultOutOfPlace() error {
        return toError(C.clfftSetResultLocation(FFTplan.clFFTHandle, C.CLFFT_OUTOFPLACE))
}

// GetResultTranspose returns true if the final transpose in the transform is skipped
func (FFTplan *ClFFTPlan) GetResultTranspose() (bool, error) {
        var val C.clfftResultTransposed
        err := toError(C.clfftGetPlanTransposeResult(FFTplan.clFFTHandle, &val))
        if err != nil {
                return false, err
        }
        switch val {
        default:
                return false, err
        case C.CLFFT_TRANSPOSED:
                return true, nil
        case C.CLFFT_NOTRANSPOSE:
                return false, nil
        }
}

func (FFTplan *ClFFTPlan) SetResultTransposed() error {
        return toError(C.clfftSetPlanTransposeResult(FFTplan.clFFTHandle, C.CLFFT_TRANSPOSED))
}

func (FFTplan *ClFFTPlan) SetResultNoTranspose() error {
        return toError(C.clfftSetPlanTransposeResult(FFTplan.clFFTHandle, C.CLFFT_NOTRANSPOSE))
}

func (FFTplan *ClFFTPlan) GetTemporaryBufferSize() (int, error) {
	var val C.size_t
	err := toError(C.clfftGetTmpBufSize(FFTplan.clFFTHandle, &val))
	if err != nil {
		return -1, err
	}
	return int(val), err
}

func (FFTplan *ClFFTPlan) SetPlanCallback(funcName string, funcString string, localMemSize int, cb_type ClFFTCallbackType, userdata *C.cl_mem, num_buffers int) error {
        cFname := make([]*C.char, 1)
        cfn := C.CString(funcName)
        cFname[0] = cfn
        cFstring := make([]*C.char, 1)
        cfs := C.CString(funcString)
        cFstring[0] = cfs
        defer C.free(unsafe.Pointer(cfn))
        defer C.free(unsafe.Pointer(cfs))

	return toError(C.clfftSetPlanCallback(FFTplan.clFFTHandle, cFname[0], cFstring[0], C.int(localMemSize),
					      C.clfftCallbackType(cb_type), userdata, C.int(num_buffers)))
}

func (FFTplan *ClFFTPlan) EnqueueForwardTransform(commQueues []*cl.CommandQueue, InWaitEventsList []*cl.Event, input_buffers_list, output_buffers_list []*cl.MemObject, tmpBufferIn *cl.MemObject) ([]C.cl_event, error) {
	return FFTplan.EnqueueTransformUnsafe(commQueues, InWaitEventsList, input_buffers_list, output_buffers_list, tmpBufferIn, ClFFTDirectionForward)
}

func (FFTplan *ClFFTPlan) EnqueueBackwardTransform(commQueues []*cl.CommandQueue, InWaitEventsList []*cl.Event, input_buffers_list, output_buffers_list []*cl.MemObject, tmpBufferIn *cl.MemObject) ([]C.cl_event, error) {
	return FFTplan.EnqueueTransformUnsafe(commQueues, InWaitEventsList, input_buffers_list, output_buffers_list, tmpBufferIn, ClFFTDirectionBackward)
}

func (FFTplan *ClFFTPlan) EnqueueTransformUnsafe(commQueues []*cl.CommandQueue, InWaitEventsList []*cl.Event, input_buffers_list, output_buffers_list []*cl.MemObject, tmpBufferIn *cl.MemObject, TransformDirection ClFFTDirection) ([]C.cl_event, error) {
	if commQueues == nil {
		panic("Null queue pointer!")
	}
	queueLength := len(commQueues)
	QueueList := make([]C.cl_command_queue, queueLength)
	for idx, id := range commQueues {
		tmp := id.GetQueueID()
		QueueList[idx] = C.FromVoidToClCommandQueue(unsafe.Pointer(&tmp))
	}

	var WaitEventsPtr *C.cl_event
	var WaitEventsList []C.cl_event
	var inWaitListLength int
	if InWaitEventsList == nil {
		WaitEventsPtr = nil
	} else {
		inWaitListLength = len(InWaitEventsList)
		WaitEventsList = make([]C.cl_event, inWaitListLength)
		for idx, id := range InWaitEventsList {
			tmp := id.ToCl()
			WaitEventsList[idx] = C.FromVoidToClEvent(unsafe.Pointer(&tmp))
		}
		WaitEventsPtr = &WaitEventsList[0]
	}

	var input_buffers_ptr *C.cl_mem
	var input_buffers []C.cl_mem
	var input_buffers_list_length int
	if input_buffers_list == nil {
		input_buffers_ptr = nil
	} else {
		input_buffers_list_length = len(input_buffers_list)
		input_buffers = make([]C.cl_mem, input_buffers_list_length)
		for idx, id := range input_buffers_list {
			tmp := id.ToCl()
			input_buffers[idx] = C.FromVoidToClMem(unsafe.Pointer(&tmp))
		}
		input_buffers_ptr = &input_buffers[0]
	}

	var output_buffers_ptr *C.cl_mem
	var output_buffers []C.cl_mem
	var output_buffers_list_length int
	if output_buffers_list == nil {
		output_buffers_ptr = nil
	} else {
		output_buffers_list_length = len(output_buffers_list)
		output_buffers = make([]C.cl_mem, output_buffers_list_length)
		for idx, id := range output_buffers_list {
			tmp := id.ToCl()
			output_buffers[idx] = C.FromVoidToClMem(unsafe.Pointer(&tmp))
		}
		output_buffers_ptr = &output_buffers[0]
	}

	var tmpBufferPtr C.cl_mem
	if tmpBufferIn != nil {
		clBuffPtr := tmpBufferIn.ToCl()
		tmpBuffer := C.FromVoidToClMem(unsafe.Pointer(&clBuffPtr))
		tmpBufferPtr = tmpBuffer
	} else {
		tmpBufferPtr = nil
	}

	outEvent := make([]C.cl_event, queueLength)
	err := toError(C.clfftEnqueueTransform(FFTplan.clFFTHandle, C.clfftDirection(TransformDirection), C.cl_uint(queueLength), &QueueList[0],
						C.cl_uint(inWaitListLength), WaitEventsPtr, &outEvent[0],
						input_buffers_ptr, output_buffers_ptr, tmpBufferPtr))
	if err != nil {
		return []C.cl_event{}, err
	}
	return outEvent, nil
}

func (ArrLayout *ArrayLayouts) SetInputLayout(layout ClFFTLayout) {
	switch layout {
	default:
		ArrLayout.inputs = CLFFTLayoutComplexInterleaved
        case CLFFTLayoutComplexInterleaved:
		ArrLayout.inputs = CLFFTLayoutComplexInterleaved
        case CLFFTLayoutComplexPlanar:
		ArrLayout.inputs = CLFFTLayoutComplexPlanar
        case CLFFTLayoutHermitianInterleaved:
		ArrLayout.inputs = CLFFTLayoutHermitianInterleaved
        case CLFFTLayoutHermitianPlanar:
		ArrLayout.inputs = CLFFTLayoutHermitianPlanar
        case CLFFTLayoutReal:
		ArrLayout.inputs = CLFFTLayoutReal
	}
}

func (ArrLayout *ArrayLayouts) SetOutputLayout(layout ClFFTLayout) {
        switch layout {
        default:
                ArrLayout.outputs = CLFFTLayoutComplexInterleaved
        case CLFFTLayoutComplexInterleaved:
                ArrLayout.outputs = CLFFTLayoutComplexInterleaved
        case CLFFTLayoutComplexPlanar:
                ArrLayout.outputs = CLFFTLayoutComplexPlanar
        case CLFFTLayoutHermitianInterleaved:
                ArrLayout.outputs = CLFFTLayoutHermitianInterleaved
        case CLFFTLayoutHermitianPlanar:
                ArrLayout.outputs = CLFFTLayoutHermitianPlanar
        case CLFFTLayoutReal:
                ArrLayout.outputs = CLFFTLayoutReal
        }
}

