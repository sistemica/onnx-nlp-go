//go:build linux && (amd64 || arm64)

package onnxgo

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
	"unsafe"
)

//go:generate glslc -fshader-stage=compute --target-env=vulkan1.0 -O shaders/gemm.comp -o shaders/gemm.spv

//go:embed shaders/gemm.spv
var gemmSPIRV []byte

const (
	vkStructApplicationInfo               = 0
	vkStructInstanceCreateInfo            = 1
	vkStructDeviceQueueCreateInfo         = 2
	vkStructDeviceCreateInfo              = 3
	vkStructSubmitInfo                    = 4
	vkStructMemoryAllocateInfo            = 5
	vkStructFenceCreateInfo               = 8
	vkStructBufferCreateInfo              = 12
	vkStructShaderModuleCreateInfo        = 16
	vkStructPipelineShaderStage           = 18
	vkStructComputePipelineCreate         = 29
	vkStructPipelineLayoutCreate          = 30
	vkStructDescriptorSetLayout           = 32
	vkStructDescriptorPoolCreate          = 33
	vkStructDescriptorSetAllocate         = 34
	vkStructWriteDescriptorSet            = 35
	vkStructCommandPoolCreateInfo         = 39
	vkStructCommandBufferAllocate         = 40
	vkStructCommandBufferBeginInfo        = 42
	vkQueueCompute                        = 0x2
	vkMemoryDeviceLocal                   = 0x1
	vkMemoryHostVisible                   = 0x2
	vkMemoryHostCoherent                  = 0x4
	vkBufferUsageStorage                  = 0x20
	vkDescriptorTypeStorageBuffer         = 7
	vkShaderStageCompute                  = 0x20
	vkPipelineBindPointCompute            = 1
	vkCommandPoolResetBuffer              = 0x2
	vkDeviceTypeIntegrated                = 1
	vkDeviceTypeDiscrete                  = 2
	vkDeviceTypeCPU                       = 4
	vkAPIVersion10                        = 1 << 22
	vkFenceTimeoutNanos            uint64 = 60 * 1000 * 1000 * 1000
	vkTileSize                            = 64
)

const (
	vkStructMemoryBarrier    = 46
	vkMemoryHostCached       = 0x8
	vkBufferUsageTransferSrc = 0x1
	vkBufferUsageTransferDst = 0x2
	vkStageCompute           = 0x800
	vkStageTransfer          = 0x1000
	vkAccessShaderRead       = 0x20
	vkAccessShaderWrite      = 0x40
	vkAccessTransferRead     = 0x800
	vkAccessTransferWrite    = 0x1000
)

type vkApplicationInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	pApplicationName   *byte
	applicationVersion uint32
	pEngineName        *byte
	engineVersion      uint32
	apiVersion         uint32
}

type vkInstanceCreateInfo struct {
	sType                   uint32
	pNext                   unsafe.Pointer
	flags                   uint32
	pApplicationInfo        *vkApplicationInfo
	enabledLayerCount       uint32
	ppEnabledLayerNames     unsafe.Pointer
	enabledExtensionCount   uint32
	ppEnabledExtensionNames unsafe.Pointer
}

type vkQueueFamilyProperties struct {
	queueFlags         uint32
	queueCount         uint32
	timestampValidBits uint32
	granularity        [3]uint32
}

type vkMemoryType struct{ propertyFlags, heapIndex uint32 }

type vkMemoryHeap struct {
	size  uint64
	flags uint32
	_     uint32
}

type vkMemoryProperties struct {
	memoryTypeCount uint32
	memoryTypes     [32]vkMemoryType
	memoryHeapCount uint32
	memoryHeaps     [16]vkMemoryHeap
}

type vkDeviceQueueCreateInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
	queueCount       uint32
	pQueuePriorities *float32
}

type vkDeviceCreateInfo struct {
	sType                   uint32
	pNext                   unsafe.Pointer
	flags                   uint32
	queueCreateInfoCount    uint32
	pQueueCreateInfos       *vkDeviceQueueCreateInfo
	enabledLayerCount       uint32
	ppEnabledLayerNames     unsafe.Pointer
	enabledExtensionCount   uint32
	ppEnabledExtensionNames unsafe.Pointer
	pEnabledFeatures        unsafe.Pointer
}

type vkShaderModuleCreateInfo struct {
	sType    uint32
	pNext    unsafe.Pointer
	flags    uint32
	codeSize uintptr
	pCode    *uint32
}

type vkDescriptorSetLayoutBinding struct {
	binding            uint32
	descriptorType     uint32
	descriptorCount    uint32
	stageFlags         uint32
	pImmutableSamplers unsafe.Pointer
}

type vkDescriptorSetLayoutCreateInfo struct {
	sType        uint32
	pNext        unsafe.Pointer
	flags        uint32
	bindingCount uint32
	pBindings    *vkDescriptorSetLayoutBinding
}

type vkPushConstantRange struct{ stageFlags, offset, size uint32 }

type vkPipelineLayoutCreateInfo struct {
	sType                  uint32
	pNext                  unsafe.Pointer
	flags                  uint32
	setLayoutCount         uint32
	pSetLayouts            *uint64
	pushConstantRangeCount uint32
	pPushConstantRanges    *vkPushConstantRange
}

type vkPipelineShaderStageCreateInfo struct {
	sType               uint32
	pNext               unsafe.Pointer
	flags               uint32
	stage               uint32
	module              uint64
	pName               *byte
	pSpecializationInfo unsafe.Pointer
}

type vkComputePipelineCreateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	flags              uint32
	stage              vkPipelineShaderStageCreateInfo
	layout             uint64
	basePipelineHandle uint64
	basePipelineIndex  int32
}

type vkDescriptorPoolSize struct{ descriptorType, descriptorCount uint32 }

type vkDescriptorPoolCreateInfo struct {
	sType         uint32
	pNext         unsafe.Pointer
	flags         uint32
	maxSets       uint32
	poolSizeCount uint32
	pPoolSizes    *vkDescriptorPoolSize
}

type vkDescriptorSetAllocateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	descriptorPool     uint64
	descriptorSetCount uint32
	pSetLayouts        *uint64
}

type vkDescriptorBufferInfo struct{ buffer, offset, rng uint64 }

type vkWriteDescriptorSet struct {
	sType            uint32
	pNext            unsafe.Pointer
	dstSet           uint64
	dstBinding       uint32
	dstArrayElement  uint32
	descriptorCount  uint32
	descriptorType   uint32
	pImageInfo       unsafe.Pointer
	pBufferInfo      *vkDescriptorBufferInfo
	pTexelBufferView unsafe.Pointer
}

type vkBufferCreateInfo struct {
	sType                 uint32
	pNext                 unsafe.Pointer
	flags                 uint32
	size                  uint64
	usage                 uint32
	sharingMode           uint32
	queueFamilyIndexCount uint32
	pQueueFamilyIndices   *uint32
}

type vkMemoryRequirements struct {
	size           uint64
	alignment      uint64
	memoryTypeBits uint32
}

type vkMemoryAllocateInfo struct {
	sType           uint32
	pNext           unsafe.Pointer
	allocationSize  uint64
	memoryTypeIndex uint32
}

type vkCommandPoolCreateInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
}

type vkCommandBufferAllocateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	commandPool        uint64
	level              uint32
	commandBufferCount uint32
}

type vkCommandBufferBeginInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	pInheritanceInfo unsafe.Pointer
}

type vkSubmitInfo struct {
	sType                uint32
	pNext                unsafe.Pointer
	waitSemaphoreCount   uint32
	pWaitSemaphores      unsafe.Pointer
	pWaitDstStageMask    unsafe.Pointer
	commandBufferCount   uint32
	pCommandBuffers      *uintptr
	signalSemaphoreCount uint32
	pSignalSemaphores    unsafe.Pointer
}

type vkFenceCreateInfo struct {
	sType uint32
	pNext unsafe.Pointer
	flags uint32
}

type vkMemoryBarrier struct {
	sType         uint32
	pNext         unsafe.Pointer
	srcAccessMask uint32
	dstAccessMask uint32
}

type vkBufferCopy struct{ srcOffset, dstOffset, size uint64 }

type vkDimensions struct{ m, k, n uint32 }

type (
	vkCreateFn  func(dev uintptr, info unsafe.Pointer, alloc uintptr, out *uint64) int32
	vkDestroyFn func(dev uintptr, handle uint64, alloc uintptr)
)

// vkAPI is the subset of Vulkan 1.0 the GEMM backend calls.
type vkAPI struct {
	createInstance           func(info unsafe.Pointer, alloc uintptr, out *uintptr) int32
	destroyInstance          func(inst uintptr, alloc uintptr)
	enumeratePhysicalDevices func(inst uintptr, count *uint32, devs *uintptr) int32
	getProperties            func(pd uintptr, out unsafe.Pointer)
	getQueueFamilies         func(pd uintptr, count *uint32, out unsafe.Pointer)
	getMemoryProperties      func(pd uintptr, out unsafe.Pointer)
	createDevice             func(pd uintptr, info unsafe.Pointer, alloc uintptr, out *uintptr) int32
	destroyDevice            func(dev uintptr, alloc uintptr)
	getDeviceQueue           func(dev uintptr, family, index uint32, out *uintptr)
	deviceWaitIdle           func(dev uintptr) int32

	createShaderModule        vkCreateFn
	createDescriptorSetLayout vkCreateFn
	createPipelineLayout      vkCreateFn
	createDescriptorPool      vkCreateFn
	createBuffer              vkCreateFn
	allocateMemory            vkCreateFn
	createCommandPool         vkCreateFn
	createFence               vkCreateFn
	createComputePipelines    func(dev uintptr, cache uint64, n uint32, infos unsafe.Pointer, alloc uintptr, out *uint64) int32

	destroyShaderModule        vkDestroyFn
	destroyDescriptorSetLayout vkDestroyFn
	destroyPipelineLayout      vkDestroyFn
	destroyDescriptorPool      vkDestroyFn
	destroyBuffer              vkDestroyFn
	freeMemory                 vkDestroyFn
	destroyCommandPool         vkDestroyFn
	destroyFence               vkDestroyFn
	destroyPipeline            vkDestroyFn

	allocateDescriptorSets func(dev uintptr, info unsafe.Pointer, out *uint64) int32
	allocateCommandBuffers func(dev uintptr, info unsafe.Pointer, out *uintptr) int32
	updateDescriptorSets   func(dev uintptr, n uint32, writes unsafe.Pointer, nCopies uint32, copies unsafe.Pointer)
	getBufferRequirements  func(dev uintptr, buf uint64, out unsafe.Pointer)
	bindBufferMemory       func(dev uintptr, buf, mem, offset uint64) int32
	mapMemory              func(dev uintptr, mem, offset, size uint64, flags uint32, out *unsafe.Pointer) int32

	beginCommandBuffer    func(cb uintptr, info unsafe.Pointer) int32
	endCommandBuffer      func(cb uintptr) int32
	resetCommandBuffer    func(cb uintptr, flags uint32) int32
	cmdBindPipeline       func(cb uintptr, point uint32, pipeline uint64)
	cmdBindDescriptorSets func(cb uintptr, point uint32, layout uint64, first, count uint32, sets *uint64, dynamicCount uint32, dynamic *uint32)
	cmdPushConstants      func(cb uintptr, layout uint64, stages, offset, size uint32, values unsafe.Pointer)
	cmdDispatch           func(cb uintptr, x, y, z uint32)
	cmdCopyBuffer         func(cb uintptr, src, dst uint64, n uint32, regions unsafe.Pointer)
	cmdPipelineBarrier    func(cb uintptr, srcStage, dstStage, flags, nMemory uint32, memory unsafe.Pointer, nBuffer uint32, buffer unsafe.Pointer, nImage uint32, image unsafe.Pointer)
	queueSubmit           func(queue uintptr, n uint32, submits unsafe.Pointer, fence uint64) int32
	waitForFences         func(dev uintptr, n uint32, fences *uint64, waitAll uint32, timeout uint64) int32
	resetFences           func(dev uintptr, n uint32, fences *uint64) int32
}

var vulkanLibraries = []string{"libvulkan.so.1", "libvulkan.so"}

func loadVulkan() (*vkAPI, error) { return loadVulkanFrom(vulkanLibraries) }

func loadVulkanFrom(names []string) (*vkAPI, error) {
	lib, _, err := dlopenFirst(names)
	if err != nil {
		return nil, fmt.Errorf("vulkan loader: %w", err)
	}
	api := &vkAPI{}
	symbols := map[string]any{
		"vkCreateInstance": &api.createInstance, "vkDestroyInstance": &api.destroyInstance,
		"vkEnumeratePhysicalDevices": &api.enumeratePhysicalDevices, "vkGetPhysicalDeviceProperties": &api.getProperties,
		"vkGetPhysicalDeviceQueueFamilyProperties": &api.getQueueFamilies, "vkGetPhysicalDeviceMemoryProperties": &api.getMemoryProperties,
		"vkCreateDevice": &api.createDevice, "vkDestroyDevice": &api.destroyDevice, "vkGetDeviceQueue": &api.getDeviceQueue,
		"vkDeviceWaitIdle": &api.deviceWaitIdle, "vkCreateShaderModule": &api.createShaderModule,
		"vkCreateDescriptorSetLayout": &api.createDescriptorSetLayout, "vkCreatePipelineLayout": &api.createPipelineLayout,
		"vkCreateDescriptorPool": &api.createDescriptorPool, "vkCreateBuffer": &api.createBuffer, "vkAllocateMemory": &api.allocateMemory,
		"vkCreateCommandPool": &api.createCommandPool, "vkCreateFence": &api.createFence, "vkCreateComputePipelines": &api.createComputePipelines,
		"vkDestroyShaderModule": &api.destroyShaderModule, "vkDestroyDescriptorSetLayout": &api.destroyDescriptorSetLayout,
		"vkDestroyPipelineLayout": &api.destroyPipelineLayout, "vkDestroyDescriptorPool": &api.destroyDescriptorPool,
		"vkDestroyBuffer": &api.destroyBuffer, "vkFreeMemory": &api.freeMemory, "vkDestroyCommandPool": &api.destroyCommandPool,
		"vkDestroyFence": &api.destroyFence, "vkDestroyPipeline": &api.destroyPipeline,
		"vkAllocateDescriptorSets": &api.allocateDescriptorSets, "vkAllocateCommandBuffers": &api.allocateCommandBuffers,
		"vkUpdateDescriptorSets": &api.updateDescriptorSets, "vkGetBufferMemoryRequirements": &api.getBufferRequirements,
		"vkBindBufferMemory": &api.bindBufferMemory, "vkMapMemory": &api.mapMemory,
		"vkBeginCommandBuffer": &api.beginCommandBuffer, "vkEndCommandBuffer": &api.endCommandBuffer,
		"vkResetCommandBuffer": &api.resetCommandBuffer, "vkCmdBindPipeline": &api.cmdBindPipeline,
		"vkCmdBindDescriptorSets": &api.cmdBindDescriptorSets, "vkCmdPushConstants": &api.cmdPushConstants,
		"vkCmdDispatch": &api.cmdDispatch, "vkCmdCopyBuffer": &api.cmdCopyBuffer, "vkCmdPipelineBarrier": &api.cmdPipelineBarrier, "vkQueueSubmit": &api.queueSubmit, "vkWaitForFences": &api.waitForFences,
		"vkResetFences": &api.resetFences,
	}
	for name, fn := range symbols {
		if err := registerFunc(fn, lib, name); err != nil {
			return nil, err
		}
	}
	return api, nil
}

func vkError(op string, code int32) error {
	return fmt.Errorf("vulkan: %s failed with code %d", op, code)
}

type vkBuffer struct {
	handle, memory, size uint64
	mapped               unsafe.Pointer
}

func (b vkBuffer) floats(n int) []float32 { return unsafe.Slice((*float32)(b.mapped), n) }

type vulkanDevice struct {
	api  *vkAPI
	name string
	mu   sync.Mutex

	instance, physical, device, queue, commandBuffer uintptr
	memory                                           vkMemoryProperties

	shader, setLayout, pipelineLayout, pipeline, pool, set, commandPool, fence uint64

	weights map[weightKey]vkBuffer // device-local copies of static weights
	staging [3]vkBuffer            // host-visible buffers for A, B and C
	local   [3]vkBuffer            // device-local buffers for A, B and C
	closed  bool
}

func init() {
	registerBackend("vulkan", openVulkan)
}

func openVulkan() (gemmDevice, error) {
	api, err := loadVulkan()
	if err != nil {
		return nil, err
	}
	return newVulkanDevice(api)
}

func cString(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func newVulkanDevice(api *vkAPI) (*vulkanDevice, error) {
	d := &vulkanDevice{api: api, weights: map[weightKey]vkBuffer{}}
	if err := d.init(); err != nil {
		d.destroy()
		return nil, err
	}
	return d, nil
}

func (d *vulkanDevice) Name() string { return d.name }

func (d *vulkanDevice) init() error {
	api := d.api
	app := vkApplicationInfo{sType: vkStructApplicationInfo, pApplicationName: cString("onnxgo"), pEngineName: cString("onnxgo"), apiVersion: vkAPIVersion10}
	info := vkInstanceCreateInfo{sType: vkStructInstanceCreateInfo, pApplicationInfo: &app}
	if code := api.createInstance(unsafe.Pointer(&info), 0, &d.instance); code != 0 {
		return vkError("vkCreateInstance", code)
	}
	physical, name, err := d.pickPhysicalDevice()
	if err != nil {
		return err
	}
	d.physical, d.name = physical, name
	api.getMemoryProperties(d.physical, unsafe.Pointer(&d.memory))
	family, err := d.computeQueueFamily()
	if err != nil {
		return err
	}
	priority := float32(1)
	queueInfo := vkDeviceQueueCreateInfo{sType: vkStructDeviceQueueCreateInfo, queueFamilyIndex: family, queueCount: 1, pQueuePriorities: &priority}
	deviceInfo := vkDeviceCreateInfo{sType: vkStructDeviceCreateInfo, queueCreateInfoCount: 1, pQueueCreateInfos: &queueInfo}
	if code := api.createDevice(d.physical, unsafe.Pointer(&deviceInfo), 0, &d.device); code != 0 {
		return vkError("vkCreateDevice", code)
	}
	api.getDeviceQueue(d.device, family, 0, &d.queue)
	return d.createPipeline(family)
}

func (d *vulkanDevice) pickPhysicalDevice() (uintptr, string, error) {
	var count uint32
	if code := d.api.enumeratePhysicalDevices(d.instance, &count, nil); code != 0 {
		return 0, "", vkError("vkEnumeratePhysicalDevices", code)
	}
	if count == 0 {
		return 0, "", errors.New("vulkan: no physical devices")
	}
	devices := make([]uintptr, count)
	if code := d.api.enumeratePhysicalDevices(d.instance, &count, &devices[0]); code != 0 {
		return 0, "", vkError("vkEnumeratePhysicalDevices", code)
	}
	want := -1
	if v, err := strconv.Atoi(os.Getenv("ONNXGO_GPU_DEVICE")); err == nil {
		want = v
	}
	allowCPU := os.Getenv("ONNXGO_VULKAN_ALLOW_CPU") == "1"
	// Integrated GPUs share memory with the CPU and are not faster than its SIMD kernel,
	// so they are only used when asked for.
	explicit := os.Getenv("ONNXGO_GPU_BACKEND") == "vulkan"
	best, bestRank, bestName := uintptr(0), -1, ""
	for i, pd := range devices[:count] {
		var props [1024]byte
		d.api.getProperties(pd, unsafe.Pointer(&props[0]))
		kind := binary.LittleEndian.Uint32(props[16:])
		name := string(props[20 : 20+indexOfNUL(props[20:276])])
		rank := map[uint32]int{vkDeviceTypeDiscrete: 3, vkDeviceTypeIntegrated: 2}[kind]
		switch {
		case want >= 0 && i == want:
			return pd, name, nil
		case want >= 0:
			continue
		case kind == vkDeviceTypeCPU && !allowCPU, kind == vkDeviceTypeIntegrated && !explicit:
			continue
		}
		if rank > bestRank {
			best, bestRank, bestName = pd, rank, name
		}
	}
	if best == 0 {
		return 0, "", errors.New("vulkan: no usable device")
	}
	return best, bestName, nil
}

func indexOfNUL(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return len(b)
}

func (d *vulkanDevice) computeQueueFamily() (uint32, error) {
	var count uint32
	d.api.getQueueFamilies(d.physical, &count, nil)
	if count == 0 {
		return 0, errors.New("vulkan: device has no queue families")
	}
	families := make([]vkQueueFamilyProperties, count)
	d.api.getQueueFamilies(d.physical, &count, unsafe.Pointer(&families[0]))
	for i, f := range families[:count] {
		if f.queueFlags&vkQueueCompute != 0 && f.queueCount > 0 {
			return uint32(i), nil
		}
	}
	return 0, errors.New("vulkan: device has no compute queue")
}

func (d *vulkanDevice) createPipeline(family uint32) error {
	api := d.api
	words := make([]uint32, len(gemmSPIRV)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(gemmSPIRV[4*i:])
	}
	shaderInfo := vkShaderModuleCreateInfo{sType: vkStructShaderModuleCreateInfo, codeSize: uintptr(len(gemmSPIRV)), pCode: &words[0]}
	if code := api.createShaderModule(d.device, unsafe.Pointer(&shaderInfo), 0, &d.shader); code != 0 {
		return vkError("vkCreateShaderModule", code)
	}
	bindings := make([]vkDescriptorSetLayoutBinding, 3)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{binding: uint32(i), descriptorType: vkDescriptorTypeStorageBuffer, descriptorCount: 1, stageFlags: vkShaderStageCompute}
	}
	layoutInfo := vkDescriptorSetLayoutCreateInfo{sType: vkStructDescriptorSetLayout, bindingCount: 3, pBindings: &bindings[0]}
	if code := api.createDescriptorSetLayout(d.device, unsafe.Pointer(&layoutInfo), 0, &d.setLayout); code != 0 {
		return vkError("vkCreateDescriptorSetLayout", code)
	}
	pushRange := vkPushConstantRange{stageFlags: vkShaderStageCompute, size: uint32(unsafe.Sizeof(vkDimensions{}))}
	pipelineLayoutInfo := vkPipelineLayoutCreateInfo{sType: vkStructPipelineLayoutCreate, setLayoutCount: 1, pSetLayouts: &d.setLayout, pushConstantRangeCount: 1, pPushConstantRanges: &pushRange}
	if code := api.createPipelineLayout(d.device, unsafe.Pointer(&pipelineLayoutInfo), 0, &d.pipelineLayout); code != 0 {
		return vkError("vkCreatePipelineLayout", code)
	}
	pipelineInfo := vkComputePipelineCreateInfo{
		sType:  vkStructComputePipelineCreate,
		stage:  vkPipelineShaderStageCreateInfo{sType: vkStructPipelineShaderStage, stage: vkShaderStageCompute, module: d.shader, pName: cString("main")},
		layout: d.pipelineLayout,
	}
	if code := api.createComputePipelines(d.device, 0, 1, unsafe.Pointer(&pipelineInfo), 0, &d.pipeline); code != 0 {
		return vkError("vkCreateComputePipelines", code)
	}
	poolSize := vkDescriptorPoolSize{descriptorType: vkDescriptorTypeStorageBuffer, descriptorCount: 3}
	poolInfo := vkDescriptorPoolCreateInfo{sType: vkStructDescriptorPoolCreate, maxSets: 1, poolSizeCount: 1, pPoolSizes: &poolSize}
	if code := api.createDescriptorPool(d.device, unsafe.Pointer(&poolInfo), 0, &d.pool); code != 0 {
		return vkError("vkCreateDescriptorPool", code)
	}
	setInfo := vkDescriptorSetAllocateInfo{sType: vkStructDescriptorSetAllocate, descriptorPool: d.pool, descriptorSetCount: 1, pSetLayouts: &d.setLayout}
	if code := api.allocateDescriptorSets(d.device, unsafe.Pointer(&setInfo), &d.set); code != 0 {
		return vkError("vkAllocateDescriptorSets", code)
	}
	cmdPoolInfo := vkCommandPoolCreateInfo{sType: vkStructCommandPoolCreateInfo, flags: vkCommandPoolResetBuffer, queueFamilyIndex: family}
	if code := api.createCommandPool(d.device, unsafe.Pointer(&cmdPoolInfo), 0, &d.commandPool); code != 0 {
		return vkError("vkCreateCommandPool", code)
	}
	cmdInfo := vkCommandBufferAllocateInfo{sType: vkStructCommandBufferAllocate, commandPool: d.commandPool, commandBufferCount: 1}
	if code := api.allocateCommandBuffers(d.device, unsafe.Pointer(&cmdInfo), &d.commandBuffer); code != 0 {
		return vkError("vkAllocateCommandBuffers", code)
	}
	fenceInfo := vkFenceCreateInfo{sType: vkStructFenceCreateInfo}
	if code := api.createFence(d.device, unsafe.Pointer(&fenceInfo), 0, &d.fence); code != 0 {
		return vkError("vkCreateFence", code)
	}
	return nil
}

func (d *vulkanDevice) destroy() {
	api := d.api
	if d.device != 0 {
		api.deviceWaitIdle(d.device)
		for _, b := range d.weights {
			d.freeBuffer(b)
		}
		for _, b := range append(d.staging[:], d.local[:]...) {
			d.freeBuffer(b)
		}
		d.weights, d.staging, d.local = map[weightKey]vkBuffer{}, [3]vkBuffer{}, [3]vkBuffer{}
		for _, h := range []struct {
			handle uint64
			fn     vkDestroyFn
		}{
			{d.fence, api.destroyFence}, {d.commandPool, api.destroyCommandPool}, {d.pool, api.destroyDescriptorPool},
			{d.pipeline, api.destroyPipeline}, {d.pipelineLayout, api.destroyPipelineLayout},
			{d.setLayout, api.destroyDescriptorSetLayout}, {d.shader, api.destroyShaderModule},
		} {
			if h.handle != 0 {
				h.fn(d.device, h.handle, 0)
			}
		}
		api.destroyDevice(d.device, 0)
	}
	if d.instance != 0 {
		api.destroyInstance(d.instance, 0)
	}
	d.device, d.instance, d.closed = 0, 0, true
}

func (d *vulkanDevice) Release() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.destroy()
}

func (d *vulkanDevice) freeBuffer(b vkBuffer) {
	if b.handle != 0 {
		d.api.destroyBuffer(d.device, b.handle, 0)
	}
	if b.memory != 0 {
		d.api.freeMemory(d.device, b.memory, 0)
	}
}

// memoryType picks a memory type allowed by typeBits. Host buffers need host-visible coherent
// memory (cached if possible); device buffers prefer device-local memory the host cannot map.
func (d *vulkanDevice) memoryType(typeBits uint32, host bool) (uint32, bool) {
	var wants [][2]uint32 // {required, forbidden}
	if host {
		coherent := uint32(vkMemoryHostVisible | vkMemoryHostCoherent)
		wants = [][2]uint32{{coherent | vkMemoryHostCached, vkMemoryDeviceLocal}, {coherent, vkMemoryDeviceLocal}, {coherent, 0}}
	} else {
		wants = [][2]uint32{{vkMemoryDeviceLocal, vkMemoryHostVisible}, {vkMemoryDeviceLocal, 0}, {0, 0}}
	}
	for _, w := range wants {
		for i := uint32(0); i < d.memory.memoryTypeCount; i++ {
			flags := d.memory.memoryTypes[i].propertyFlags
			if typeBits&(1<<i) != 0 && flags&w[0] == w[0] && flags&w[1] == 0 {
				return i, true
			}
		}
	}
	return 0, false
}

func (d *vulkanDevice) newBuffer(size uint64, host bool) (vkBuffer, error) {
	api := d.api
	var b vkBuffer
	b.size = size
	info := vkBufferCreateInfo{sType: vkStructBufferCreateInfo, size: size, usage: vkBufferUsageStorage | vkBufferUsageTransferSrc | vkBufferUsageTransferDst}
	if code := api.createBuffer(d.device, unsafe.Pointer(&info), 0, &b.handle); code != 0 {
		return vkBuffer{}, vkError("vkCreateBuffer", code)
	}
	var req vkMemoryRequirements
	api.getBufferRequirements(d.device, b.handle, unsafe.Pointer(&req))
	typeIndex, ok := d.memoryType(req.memoryTypeBits, host)
	if !ok {
		d.freeBuffer(b)
		return vkBuffer{}, errors.New("vulkan: no suitable memory type")
	}
	alloc := vkMemoryAllocateInfo{sType: vkStructMemoryAllocateInfo, allocationSize: req.size, memoryTypeIndex: typeIndex}
	if code := api.allocateMemory(d.device, unsafe.Pointer(&alloc), 0, &b.memory); code != 0 {
		d.freeBuffer(vkBuffer{handle: b.handle})
		return vkBuffer{}, vkError("vkAllocateMemory", code)
	}
	if code := api.bindBufferMemory(d.device, b.handle, b.memory, 0); code != 0 {
		d.freeBuffer(b)
		return vkBuffer{}, vkError("vkBindBufferMemory", code)
	}
	if host {
		if code := api.mapMemory(d.device, b.memory, 0, size, 0, &b.mapped); code != 0 {
			d.freeBuffer(b)
			return vkBuffer{}, vkError("vkMapMemory", code)
		}
	}
	return b, nil
}

// grown returns slot's buffer, reallocating it when it is smaller than size.
func (d *vulkanDevice) grown(slots *[3]vkBuffer, slot int, size uint64, host bool) (vkBuffer, error) {
	if slots[slot].size >= size {
		return slots[slot], nil
	}
	d.freeBuffer(slots[slot])
	slots[slot] = vkBuffer{}
	b, err := d.newBuffer(size, host)
	if err != nil {
		return vkBuffer{}, err
	}
	slots[slot] = b
	return b, nil
}

// submit records commands into the shared command buffer, runs them and waits for completion.
func (d *vulkanDevice) submit(record func(cb uintptr)) error {
	api := d.api
	if code := api.resetCommandBuffer(d.commandBuffer, 0); code != 0 {
		return vkError("vkResetCommandBuffer", code)
	}
	begin := vkCommandBufferBeginInfo{sType: vkStructCommandBufferBeginInfo}
	if code := api.beginCommandBuffer(d.commandBuffer, unsafe.Pointer(&begin)); code != 0 {
		return vkError("vkBeginCommandBuffer", code)
	}
	record(d.commandBuffer)
	if code := api.endCommandBuffer(d.commandBuffer); code != 0 {
		return vkError("vkEndCommandBuffer", code)
	}
	if code := api.resetFences(d.device, 1, &d.fence); code != 0 {
		return vkError("vkResetFences", code)
	}
	info := vkSubmitInfo{sType: vkStructSubmitInfo, commandBufferCount: 1, pCommandBuffers: &d.commandBuffer}
	if code := api.queueSubmit(d.queue, 1, unsafe.Pointer(&info), d.fence); code != 0 {
		return vkError("vkQueueSubmit", code)
	}
	if code := api.waitForFences(d.device, 1, &d.fence, 1, vkFenceTimeoutNanos); code != 0 {
		return vkError("vkWaitForFences", code)
	}
	return nil
}

func (d *vulkanDevice) copyBuffer(cb uintptr, src, dst vkBuffer, size uint64) {
	region := vkBufferCopy{size: size}
	d.api.cmdCopyBuffer(cb, src.handle, dst.handle, 1, unsafe.Pointer(&region))
}

func (d *vulkanDevice) barrier(cb uintptr, srcStage, dstStage, srcAccess, dstAccess uint32) {
	mem := vkMemoryBarrier{sType: vkStructMemoryBarrier, srcAccessMask: srcAccess, dstAccessMask: dstAccess}
	d.api.cmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, unsafe.Pointer(&mem), 0, nil, 0, nil)
}

// weightBuffer returns the device-local copy of a static matrix, uploading it on first use.
func (d *vulkanDevice) weightBuffer(j gemmJob) (vkBuffer, bool) {
	key := weightKey{&j.b[0], len(j.b)}
	if b, ok := d.weights[key]; ok {
		return b, true
	}
	bytes := uint64(len(j.b)) * 4
	local, err := d.newBuffer(bytes, false)
	if err != nil {
		return vkBuffer{}, false
	}
	staging, err := d.newBuffer(bytes, true)
	if err != nil {
		d.freeBuffer(local)
		return vkBuffer{}, false
	}
	defer d.freeBuffer(staging)
	copy(staging.floats(len(j.b)), j.b)
	if err := d.submit(func(cb uintptr) { d.copyBuffer(cb, staging, local, bytes) }); err != nil {
		d.freeBuffer(local)
		return vkBuffer{}, false
	}
	d.weights[key] = local
	return local, true
}

func (d *vulkanDevice) Gemm(j gemmJob) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return errors.New("vulkan: device released")
	}
	aBytes, bBytes, cBytes := uint64(len(j.a))*4, uint64(len(j.b))*4, uint64(len(j.out))*4

	var devB vkBuffer
	uploadB := true
	if j.bStatic {
		devB, uploadB = d.weightBuffer(j)
		uploadB = !uploadB
	}
	var stagedB vkBuffer
	var err error
	if uploadB {
		if stagedB, err = d.grown(&d.staging, 1, bBytes, true); err != nil {
			return err
		}
		if devB, err = d.grown(&d.local, 1, bBytes, false); err != nil {
			return err
		}
		copy(stagedB.floats(len(j.b)), j.b)
	}
	stagedA, err := d.grown(&d.staging, 0, aBytes, true)
	if err != nil {
		return err
	}
	devA, err := d.grown(&d.local, 0, aBytes, false)
	if err != nil {
		return err
	}
	stagedC, err := d.grown(&d.staging, 2, cBytes, true)
	if err != nil {
		return err
	}
	devC, err := d.grown(&d.local, 2, cBytes, false)
	if err != nil {
		return err
	}
	copy(stagedA.floats(len(j.a)), j.a)
	d.bindBuffers(devA, devB, devC)

	dims := vkDimensions{uint32(j.m), uint32(j.k), uint32(j.n)}
	err = d.submit(func(cb uintptr) {
		d.copyBuffer(cb, stagedA, devA, aBytes)
		if uploadB {
			d.copyBuffer(cb, stagedB, devB, bBytes)
		}
		d.barrier(cb, vkStageTransfer, vkStageCompute, vkAccessTransferWrite, vkAccessShaderRead)
		d.api.cmdBindPipeline(cb, vkPipelineBindPointCompute, d.pipeline)
		d.api.cmdBindDescriptorSets(cb, vkPipelineBindPointCompute, d.pipelineLayout, 0, 1, &d.set, 0, nil)
		d.api.cmdPushConstants(cb, d.pipelineLayout, vkShaderStageCompute, 0, uint32(unsafe.Sizeof(dims)), unsafe.Pointer(&dims))
		d.api.cmdDispatch(cb, (dims.n+vkTileSize-1)/vkTileSize, (dims.m+vkTileSize-1)/vkTileSize, 1)
		d.barrier(cb, vkStageCompute, vkStageTransfer, vkAccessShaderWrite, vkAccessTransferRead)
		d.copyBuffer(cb, devC, stagedC, cBytes)
	})
	if err != nil {
		return err
	}
	copy(j.out, stagedC.floats(len(j.out)))
	return nil
}

func (d *vulkanDevice) bindBuffers(a, b, c vkBuffer) {
	buffers := [3]vkDescriptorBufferInfo{{a.handle, 0, a.size}, {b.handle, 0, b.size}, {c.handle, 0, c.size}}
	var writes [3]vkWriteDescriptorSet
	for i := range writes {
		writes[i] = vkWriteDescriptorSet{sType: vkStructWriteDescriptorSet, dstSet: d.set, dstBinding: uint32(i), descriptorCount: 1, descriptorType: vkDescriptorTypeStorageBuffer, pBufferInfo: &buffers[i]}
	}
	d.api.updateDescriptorSets(d.device, 3, unsafe.Pointer(&writes[0]), 0, nil)
}
