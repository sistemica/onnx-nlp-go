//go:build linux && (amd64 || arm64)

package onnxgo

import (
	"encoding/binary"
	"math/rand/v2"
	"strings"
	"testing"
	"unsafe"
)

type fakeCommand struct {
	kind     string
	src, dst uint64
	size     uint64
	dims     vkDimensions
}

type fakeVulkanDevice struct {
	name       string
	kind       uint32
	families   []vkQueueFamilyProperties
	memTypes   []vkMemoryType
	typeBitsOf func(host bool) uint32
}

type fakeVulkan struct {
	devices  []fakeVulkanDevice
	failures map[string]int32
	next     uint64
	created  map[uint64]string
	freed    int
	buffers  map[uint64]*fakeVkBuffer
	memory   map[uint64][]byte
	set      [3]uint64
	dims     vkDimensions
	recorded []fakeCommand
	submits  int
	chosen   uintptr
	instance bool
	device   bool
}

type fakeVkBuffer struct {
	size   uint64
	memory uint64
}

func standardMemory() []vkMemoryType {
	return []vkMemoryType{
		{vkMemoryDeviceLocal, 0},
		{vkMemoryHostVisible | vkMemoryHostCoherent, 1},
		{vkMemoryHostVisible | vkMemoryHostCoherent | vkMemoryHostCached, 1},
		{vkMemoryDeviceLocal | vkMemoryHostVisible | vkMemoryHostCoherent, 0},
	}
}

func gpuDevice(name string, kind uint32) fakeVulkanDevice {
	return fakeVulkanDevice{name: name, kind: kind, families: []vkQueueFamilyProperties{{queueFlags: 1, queueCount: 1}, {queueFlags: vkQueueCompute, queueCount: 2}}, memTypes: standardMemory()}
}

func newFakeVulkan(devices ...fakeVulkanDevice) *fakeVulkan {
	return &fakeVulkan{devices: devices, failures: map[string]int32{}, next: 0x100, created: map[uint64]string{}, buffers: map[uint64]*fakeVkBuffer{}, memory: map[uint64][]byte{}}
}

func (f *fakeVulkan) handle(kind string) uint64 {
	f.next += 8
	f.created[f.next] = kind
	return f.next
}

func (f *fakeVulkan) creator(name string) vkCreateFn {
	return func(dev uintptr, info unsafe.Pointer, alloc uintptr, out *uint64) int32 {
		if code := f.failures[name]; code != 0 {
			return code
		}
		*out = f.handle(name)
		switch name {
		case "vkCreateBuffer":
			f.buffers[*out] = &fakeVkBuffer{size: (*vkBufferCreateInfo)(info).size}
		case "vkAllocateMemory":
			f.memory[*out] = make([]byte, (*vkMemoryAllocateInfo)(info).allocationSize+8)
		}
		return 0
	}
}

func (f *fakeVulkan) destroyer() vkDestroyFn {
	return func(dev uintptr, handle uint64, alloc uintptr) {
		delete(f.created, handle)
		f.freed++
	}
}

func (f *fakeVulkan) typeBits() uint32 { return 1<<len(f.devices[f.chosen].memTypes) - 1 }

func (f *fakeVulkan) api() *vkAPI {
	fail := func(name string) int32 { return f.failures[name] }
	return &vkAPI{
		createInstance: func(info unsafe.Pointer, alloc uintptr, out *uintptr) int32 {
			*out = 0xA1
			f.instance = true
			return fail("vkCreateInstance")
		},
		destroyInstance: func(uintptr, uintptr) { f.instance = false },
		enumeratePhysicalDevices: func(inst uintptr, count *uint32, devs *uintptr) int32 {
			if code := fail("vkEnumeratePhysicalDevices"); code != 0 {
				return code
			}
			*count = uint32(len(f.devices))
			if devs != nil {
				all := unsafe.Slice(devs, len(f.devices))
				for i := range all {
					all[i] = uintptr(i + 1)
				}
			}
			return 0
		},
		getProperties: func(pd uintptr, out unsafe.Pointer) {
			raw := unsafe.Slice((*byte)(out), 1024)
			dev := f.devices[pd-1]
			binary.LittleEndian.PutUint32(raw[16:], dev.kind)
			copy(raw[20:], dev.name+"\x00")
		},
		getQueueFamilies: func(pd uintptr, count *uint32, out unsafe.Pointer) {
			fams := f.devices[pd-1].families
			if out == nil {
				*count = uint32(len(fams))
				return
			}
			copy(unsafe.Slice((*vkQueueFamilyProperties)(out), len(fams)), fams)
		},
		getMemoryProperties: func(pd uintptr, out unsafe.Pointer) {
			f.chosen = pd - 1
			props := (*vkMemoryProperties)(out)
			types := f.devices[pd-1].memTypes
			props.memoryTypeCount = uint32(len(types))
			copy(props.memoryTypes[:], types)
		},
		createDevice: func(pd uintptr, info unsafe.Pointer, alloc uintptr, out *uintptr) int32 {
			*out = 0xB2
			f.device = true
			return fail("vkCreateDevice")
		},
		destroyDevice:  func(uintptr, uintptr) { f.device = false },
		getDeviceQueue: func(dev uintptr, family, index uint32, out *uintptr) { *out = 0xC3 },
		deviceWaitIdle: func(uintptr) int32 { return 0 },

		createShaderModule:        f.creator("vkCreateShaderModule"),
		createDescriptorSetLayout: f.creator("vkCreateDescriptorSetLayout"),
		createPipelineLayout:      f.creator("vkCreatePipelineLayout"),
		createDescriptorPool:      f.creator("vkCreateDescriptorPool"),
		createBuffer:              f.creator("vkCreateBuffer"),
		allocateMemory:            f.creator("vkAllocateMemory"),
		createCommandPool:         f.creator("vkCreateCommandPool"),
		createFence:               f.creator("vkCreateFence"),
		createComputePipelines: func(dev uintptr, cache uint64, n uint32, infos unsafe.Pointer, alloc uintptr, out *uint64) int32 {
			return f.creator("vkCreateComputePipelines")(dev, infos, alloc, out)
		},
		destroyShaderModule: f.destroyer(), destroyDescriptorSetLayout: f.destroyer(), destroyPipelineLayout: f.destroyer(),
		destroyDescriptorPool: f.destroyer(), destroyBuffer: f.destroyer(), freeMemory: f.destroyer(),
		destroyCommandPool: f.destroyer(), destroyFence: f.destroyer(), destroyPipeline: f.destroyer(),

		allocateDescriptorSets: func(dev uintptr, info unsafe.Pointer, out *uint64) int32 {
			*out = 0xD4
			return fail("vkAllocateDescriptorSets")
		},
		allocateCommandBuffers: func(dev uintptr, info unsafe.Pointer, out *uintptr) int32 {
			*out = 0xE5
			return fail("vkAllocateCommandBuffers")
		},
		updateDescriptorSets: func(dev uintptr, n uint32, writes unsafe.Pointer, nc uint32, copies unsafe.Pointer) {
			for i, w := range unsafe.Slice((*vkWriteDescriptorSet)(writes), n) {
				f.set[i] = w.pBufferInfo.buffer
			}
		},
		getBufferRequirements: func(dev uintptr, buf uint64, out unsafe.Pointer) {
			req := (*vkMemoryRequirements)(out)
			req.size = f.buffers[buf].size
			req.memoryTypeBits = f.typeBits()
			if bits := f.failures["typebits"]; bits != 0 {
				req.memoryTypeBits = uint32(bits) & 0x0
			}
		},
		bindBufferMemory: func(dev uintptr, buf, mem, offset uint64) int32 {
			if code := fail("vkBindBufferMemory"); code != 0 {
				return code
			}
			f.buffers[buf].memory = mem
			return 0
		},
		mapMemory: func(dev uintptr, mem, offset, size uint64, flags uint32, out *unsafe.Pointer) int32 {
			if code := fail("vkMapMemory"); code != 0 {
				return code
			}
			*out = unsafe.Pointer(&f.memory[mem][0])
			return 0
		},

		beginCommandBuffer: func(cb uintptr, info unsafe.Pointer) int32 { f.recorded = nil; return fail("vkBeginCommandBuffer") },
		endCommandBuffer:   func(uintptr) int32 { return fail("vkEndCommandBuffer") },
		resetCommandBuffer: func(uintptr, uint32) int32 { return fail("vkResetCommandBuffer") },
		cmdBindPipeline:    func(uintptr, uint32, uint64) {},
		cmdBindDescriptorSets: func(cb uintptr, point uint32, layout uint64, first, count uint32, sets *uint64, dc uint32, dyn *uint32) {
		},
		cmdPushConstants: func(cb uintptr, layout uint64, stages, offset, size uint32, values unsafe.Pointer) {
			f.dims = *(*vkDimensions)(values)
		},
		cmdDispatch: func(cb uintptr, x, y, z uint32) {
			f.recorded = append(f.recorded, fakeCommand{kind: "dispatch", dims: f.dims})
		},
		cmdCopyBuffer: func(cb uintptr, src, dst uint64, n uint32, regions unsafe.Pointer) {
			f.recorded = append(f.recorded, fakeCommand{kind: "copy", src: src, dst: dst, size: (*vkBufferCopy)(regions).size})
		},
		cmdPipelineBarrier: func(cb uintptr, srcStage, dstStage, flags, nMem uint32, mem unsafe.Pointer, nBuf uint32, buf unsafe.Pointer, nImg uint32, img unsafe.Pointer) {
			f.recorded = append(f.recorded, fakeCommand{kind: "barrier"})
		},
		queueSubmit: func(queue uintptr, n uint32, submits unsafe.Pointer, fence uint64) int32 {
			if code := fail("vkQueueSubmit"); code != 0 {
				return code
			}
			f.submits++
			f.execute()
			return 0
		},
		waitForFences: func(dev uintptr, n uint32, fences *uint64, all uint32, timeout uint64) int32 {
			return fail("vkWaitForFences")
		},
		resetFences: func(dev uintptr, n uint32, fences *uint64) int32 { return fail("vkResetFences") },
	}
}

func (f *fakeVulkan) bytesOf(buf uint64) []byte { return f.memory[f.buffers[buf].memory] }

func (f *fakeVulkan) execute() {
	for _, c := range f.recorded {
		switch c.kind {
		case "copy":
			copy(f.bytesOf(c.dst)[:c.size], f.bytesOf(c.src)[:c.size])
		case "dispatch":
			a, b, out := floatsOf(f.bytesOf(f.set[0])), floatsOf(f.bytesOf(f.set[1])), floatsOf(f.bytesOf(f.set[2]))
			m, k, n := int(c.dims.m), int(c.dims.k), int(c.dims.n)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var sum float32
					for p := 0; p < k; p++ {
						sum += a[i*k+p] * b[p*n+j]
					}
					out[i*n+j] = sum
				}
			}
		}
	}
}

func discrete() fakeVulkanDevice { return gpuDevice("Fake Discrete", vkDeviceTypeDiscrete) }

func TestFakeVulkanGemmRunsTheCommandSequence(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeVulkan(discrete())
	dev, err := newVulkanDevice(drv.api())
	if err != nil {
		t.Fatal(err)
	}
	if dev.Name() != "Fake Discrete" {
		t.Fatalf("name %q", dev.Name())
	}
	rng := rand.New(rand.NewPCG(71, 72))
	for _, sz := range [][3]int{{3, 4, 5}, {17, 9, 33}, {2, 2, 2}} {
		for _, static := range []bool{false, true} {
			job, want := fakeJob(rng, sz[0], sz[1], sz[2], static)
			for round := 0; round < 2; round++ {
				clear(job.out)
				if err := dev.Gemm(job); err != nil {
					t.Fatal(err)
				}
				closeSlices(t, job.out, want)
			}
		}
	}
	if len(dev.weights) != 3 {
		t.Fatalf("expected one cached weight matrix per static job, got %d", len(dev.weights))
	}
	dev.Release()
	dev.Release()
	if len(drv.created) != 0 || drv.instance || drv.device {
		t.Fatalf("release left %d handles, instance=%v device=%v", len(drv.created), drv.instance, drv.device)
	}
	rngJob, _ := fakeJob(rng, 2, 2, 2, false)
	if err := dev.Gemm(rngJob); err == nil || !strings.Contains(err.Error(), "released") {
		t.Fatalf("use after release: %v", err)
	}
}

func TestFakeVulkanBarriersSurroundTheDispatch(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeVulkan(discrete())
	dev, _ := newVulkanDevice(drv.api())
	rng := rand.New(rand.NewPCG(73, 74))
	job, _ := fakeJob(rng, 4, 4, 4, false)
	if err := dev.Gemm(job); err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, c := range drv.recorded {
		kinds = append(kinds, c.kind)
	}
	if got := strings.Join(kinds, ","); got != "copy,copy,barrier,dispatch,barrier,copy" {
		t.Fatalf("command order %s", got)
	}
	dev.Release()
}

func TestFakeVulkanDeviceSelection(t *testing.T) {
	isolateGPUEnv(t)
	devices := []fakeVulkanDevice{
		gpuDevice("Software", vkDeviceTypeCPU),
		gpuDevice("Integrated", vkDeviceTypeIntegrated),
		gpuDevice("Discrete", vkDeviceTypeDiscrete),
		gpuDevice("Other", 0),
	}
	pick := func(env map[string]string) (string, error) {
		for k, v := range env {
			t.Setenv(k, v)
		}
		dev, err := newVulkanDevice(newFakeVulkan(devices...).api())
		if err != nil {
			return "", err
		}
		defer dev.Release()
		return dev.Name(), nil
	}
	clear := map[string]string{"ONNXGO_GPU_DEVICE": "", "ONNXGO_GPU_BACKEND": "", "ONNXGO_VULKAN_ALLOW_CPU": ""}
	if name, err := pick(clear); err != nil || name != "Discrete" {
		t.Fatalf("default: %q %v", name, err)
	}
	if name, err := pick(map[string]string{"ONNXGO_GPU_DEVICE": "1"}); err != nil || name != "Integrated" {
		t.Fatalf("explicit index: %q %v", name, err)
	}
	if name, err := pick(map[string]string{"ONNXGO_GPU_DEVICE": "0"}); err != nil || name != "Software" {
		t.Fatalf("explicit index may select any device: %q %v", name, err)
	}
	if _, err := pick(map[string]string{"ONNXGO_GPU_DEVICE": "9"}); err == nil {
		t.Fatal("an out-of-range index must fail")
	}
	devices = devices[:2]
	if _, err := pick(clear); err == nil || !strings.Contains(err.Error(), "no usable device") {
		t.Fatalf("software and integrated devices must be skipped by default: %v", err)
	}
	if name, err := pick(map[string]string{"ONNXGO_GPU_BACKEND": "vulkan"}); err != nil || name != "Integrated" {
		t.Fatalf("integrated devices are allowed when the backend is requested: %q %v", name, err)
	}
	devices = devices[:1]
	if name, err := pick(map[string]string{"ONNXGO_GPU_BACKEND": "", "ONNXGO_VULKAN_ALLOW_CPU": "1"}); err != nil || name != "Software" {
		t.Fatalf("software device on request: %q %v", name, err)
	}
}

func TestFakeVulkanInitializationFailures(t *testing.T) {
	isolateGPUEnv(t)
	t.Setenv("ONNXGO_GPU_DEVICE", "")
	for _, name := range []string{
		"vkCreateInstance", "vkEnumeratePhysicalDevices", "vkCreateDevice", "vkCreateShaderModule", "vkCreateDescriptorSetLayout",
		"vkCreatePipelineLayout", "vkCreateComputePipelines", "vkCreateDescriptorPool", "vkAllocateDescriptorSets",
		"vkCreateCommandPool", "vkAllocateCommandBuffers", "vkCreateFence",
	} {
		drv := newFakeVulkan(discrete())
		drv.failures[name] = 5
		if _, err := newVulkanDevice(drv.api()); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: error %v", name, err)
		}
		if len(drv.created) != 0 || drv.instance || drv.device {
			t.Errorf("%s: failed initialization leaked %d handles (instance=%v device=%v)", name, len(drv.created), drv.instance, drv.device)
		}
	}
	if _, err := newVulkanDevice(newFakeVulkan().api()); err == nil {
		t.Fatal("no physical devices must fail")
	}
	noCompute := discrete()
	noCompute.families = []vkQueueFamilyProperties{{queueFlags: 1, queueCount: 1}, {queueFlags: vkQueueCompute, queueCount: 0}}
	if _, err := newVulkanDevice(newFakeVulkan(noCompute).api()); err == nil || !strings.Contains(err.Error(), "no compute queue") {
		t.Fatalf("no compute queue: %v", err)
	}
	noFamilies := discrete()
	noFamilies.families = nil
	if _, err := newVulkanDevice(newFakeVulkan(noFamilies).api()); err == nil || !strings.Contains(err.Error(), "no queue families") {
		t.Fatalf("no queue families: %v", err)
	}
	if err := vkError("vkQueueSubmit", -4); !strings.Contains(err.Error(), "vkQueueSubmit") || !strings.Contains(err.Error(), "-4") {
		t.Fatalf("error text %q", err)
	}
}

func TestFakeVulkanGemmFailures(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(75, 76))
	for _, name := range []string{
		"vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory", "vkMapMemory", "vkResetCommandBuffer", "vkBeginCommandBuffer",
		"vkEndCommandBuffer", "vkResetFences", "vkQueueSubmit", "vkWaitForFences",
	} {
		for _, static := range []bool{false, true} {
			drv := newFakeVulkan(discrete())
			dev, err := newVulkanDevice(drv.api())
			if err != nil {
				t.Fatal(err)
			}
			drv.failures[name] = 7
			job, want := fakeJob(rng, 3, 4, 5, static)
			err = dev.Gemm(job)
			if static && err == nil {
				closeSlices(t, job.out, want)
				continue
			}
			if err == nil {
				t.Errorf("%s (static=%v): failure was ignored", name, static)
			}
			delete(drv.failures, name)
			dev.Release()
			if len(drv.created) != 0 {
				t.Errorf("%s (static=%v): %d handles leaked", name, static, len(drv.created))
			}
		}
	}
}

func TestFakeVulkanFailsOnEachSuccessiveBuffer(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(77, 78))
	for failAt := 1; failAt <= 8; failAt++ {
		drv := newFakeVulkan(discrete())
		dev, _ := newVulkanDevice(drv.api())
		api := dev.api
		count := 0
		inner := api.createBuffer
		api.createBuffer = func(d uintptr, info unsafe.Pointer, alloc uintptr, out *uint64) int32 {
			if count++; count == failAt {
				return 9
			}
			return inner(d, info, alloc, out)
		}
		job, _ := fakeJob(rng, 3, 4, 5, false)
		if err := dev.Gemm(job); err == nil && failAt <= 6 {
			t.Errorf("buffer %d: failure was ignored", failAt)
		}
		dev.Release()
	}
}

func TestFakeVulkanMemoryTypeSelection(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(79, 80))
	layouts := map[string][]vkMemoryType{
		"unified memory only":     {{vkMemoryDeviceLocal | vkMemoryHostVisible | vkMemoryHostCoherent, 0}},
		"host memory not cached":  {{vkMemoryDeviceLocal, 0}, {vkMemoryHostVisible | vkMemoryHostCoherent, 1}},
		"no pure device memory":   {{vkMemoryDeviceLocal | vkMemoryHostVisible | vkMemoryHostCoherent, 0}, {vkMemoryHostVisible | vkMemoryHostCoherent, 1}},
		"only generic memory":     {{vkMemoryHostVisible | vkMemoryHostCoherent, 0}},
		"device memory not local": {{vkMemoryHostVisible | vkMemoryHostCoherent, 0}, {0, 1}},
	}
	for name, types := range layouts {
		dev := discrete()
		dev.memTypes = types
		drv := newFakeVulkan(dev)
		d, err := newVulkanDevice(drv.api())
		if err != nil {
			t.Fatal(err)
		}
		job, want := fakeJob(rng, 4, 4, 4, false)
		if err := d.Gemm(job); err != nil {
			t.Errorf("%s: %v", name, err)
		} else {
			closeSlices(t, job.out, want)
		}
		d.Release()
	}
	dev := discrete()
	dev.memTypes = []vkMemoryType{{vkMemoryDeviceLocal, 0}}
	d, _ := newVulkanDevice(newFakeVulkan(dev).api())
	job, _ := fakeJob(rng, 2, 2, 2, false)
	if err := d.Gemm(job); err == nil || !strings.Contains(err.Error(), "no suitable memory type") {
		t.Fatalf("no host-visible memory: %v", err)
	}
	d.Release()
}

func TestFakeVulkanStaticUploadFallsBackToStaging(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(81, 82))
	for _, failing := range []string{"vkQueueSubmit", "vkCreateBuffer"} {
		drv := newFakeVulkan(discrete())
		dev, _ := newVulkanDevice(drv.api())
		api := dev.api
		calls := 0
		switch failing {
		case "vkQueueSubmit":
			inner := api.queueSubmit
			api.queueSubmit = func(q uintptr, n uint32, s unsafe.Pointer, fence uint64) int32 {
				if calls++; calls == 1 {
					return 3
				}
				return inner(q, n, s, fence)
			}
		default:
			inner := api.createBuffer
			api.createBuffer = func(d uintptr, info unsafe.Pointer, alloc uintptr, out *uint64) int32 {
				if calls++; calls == 1 {
					return 3
				}
				return inner(d, info, alloc, out)
			}
		}
		job, want := fakeJob(rng, 3, 4, 5, true)
		if err := dev.Gemm(job); err != nil {
			t.Fatalf("%s: %v", failing, err)
		}
		closeSlices(t, job.out, want)
		if len(dev.weights) != 0 {
			t.Fatalf("%s: a failed weight upload must not be cached", failing)
		}
		dev.Release()
	}
}

func TestFakeVulkanStagingFailureAfterDeviceBufferFreesIt(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(83, 84))
	drv := newFakeVulkan(discrete())
	dev, _ := newVulkanDevice(drv.api())
	api := dev.api
	calls := 0
	inner := api.createBuffer
	api.createBuffer = func(d uintptr, info unsafe.Pointer, alloc uintptr, out *uint64) int32 {
		if calls++; calls == 2 {
			return 3
		}
		return inner(d, info, alloc, out)
	}
	job, want := fakeJob(rng, 3, 4, 5, true)
	if err := dev.Gemm(job); err != nil {
		t.Fatal(err)
	}
	closeSlices(t, job.out, want)
	dev.Release()
	if len(drv.created) != 0 {
		t.Fatalf("%d handles leaked", len(drv.created))
	}
}

func TestLoadVulkanReportsMissingLibraries(t *testing.T) {
	isolateGPUEnv(t)
	if _, err := loadVulkanFrom([]string{"libdoesnotexist.so.99"}); err == nil || !strings.Contains(err.Error(), "vulkan loader") {
		t.Fatalf("missing loader: %v", err)
	}
	if _, err := loadVulkanFrom([]string{"libc.so.6"}); err == nil || !strings.Contains(err.Error(), "symbol ") {
		t.Fatalf("a library without Vulkan symbols must name a missing symbol: %v", err)
	}
	if _, err := loadVulkan(); err != nil {
		t.Logf("no Vulkan loader here: %v", err)
	}
	if _, err := openVulkan(); err != nil {
		t.Logf("no Vulkan device here: %v", err)
	}
}

func TestOpenVulkanWithoutLoader(t *testing.T) {
	isolateGPUEnv(t)
	saved := vulkanLibraries
	defer func() { vulkanLibraries = saved }()
	vulkanLibraries = []string{"libdoesnotexist.so.99"}
	if _, err := openVulkan(); err == nil || !strings.Contains(err.Error(), "vulkan loader") {
		t.Fatalf("missing loader: %v", err)
	}
}

func TestFakeVulkanEnumerationFailsOnSecondCall(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeVulkan(discrete())
	api := drv.api()
	calls := 0
	inner := api.enumeratePhysicalDevices
	api.enumeratePhysicalDevices = func(inst uintptr, count *uint32, devs *uintptr) int32 {
		if calls++; calls == 2 {
			return 6
		}
		return inner(inst, count, devs)
	}
	if _, err := newVulkanDevice(api); err == nil || !strings.Contains(err.Error(), "vkEnumeratePhysicalDevices") {
		t.Fatalf("second enumeration failure: %v", err)
	}
}

func TestFakeVulkanDeviceNameWithoutTerminator(t *testing.T) {
	isolateGPUEnv(t)
	t.Setenv("ONNXGO_GPU_DEVICE", "")
	long := strings.Repeat("x", 300)
	dev, err := newVulkanDevice(newFakeVulkan(gpuDevice(long, vkDeviceTypeDiscrete)).api())
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Release()
	if len(dev.Name()) != 256 {
		t.Fatalf("name length %d, want the 256-byte field", len(dev.Name()))
	}
}
