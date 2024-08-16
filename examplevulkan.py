import time
import asyncio
import glm
import numpy as np
import vulkan as vk
from cffi import FFI

ffi = FFI()

# Custom visualization class for drawing a circle
class ExVkCircle:
    def __init__(self, width, height, color):
        # private app variables
        self._width = width
        self._height = height
        self._center = [self._width // 2, self._height // 2]
        self._color = color
        self._radius = 100
        self._thickness = 4
        self._velocity_x = 100
        self._velocity_y = 60
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}
        self._gpu_video_encode = False
        self._start_time = round(time.time_ns() / 1000000)
        self._prev_time = self._start_time
        # private vulkan variables
        self._instance = None
        self._physical_device = None
        self._device_mem_props = None
        self._logical_device = None
        self._queue_family_index = -1
        self._graphic_queue = None
        self._semaphore_image_available = None
        self._semaphore_render_finished = None
        self._command_pool = None
        self._vertex_object = {"vertices": None, "indices": None}
        self._vertex_binding_desc = None
        self._vertex_attribs_desc = None
        self._uniforms = {}
        self._color_format = 0
        self._color_attachment = {}
        self._color_image_view = None
        self._depth_format = 0
        self._depth_attachment = {}
        self._depth_image_view = None
        self._framebuffer = None
        
        # initialie
        self._initVulkan()

    def _initVulkan(self):
        self._createInstance()
        self._findPhysicalDevice()
        self._checkVideoEncodingSupport()
        self._findQueueFamilies()
        self._createLogicalDevice()
        self._createSemaphores()
        self._createCommandPool()
        self._createVertexBuffer()
        self._createUniformBuffer()
        self._createImageViews()
        self._createRenderPass()
        self._createFramebuffers()
        self._createGraphicsPipeline()
        #self._createDescriptorPool()
        #self._createDescriptorSet()
        #self._createCommandBuffers()
    
    def _createInstance(self):
        # Vulkan: application information
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Hello Triangle",
            applicationVersion=vk.VK_MAKE_VERSION(1, 3, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 3, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0)
        )

        # Vulkan: instance layers and extensions
        layers = vk.vkEnumerateInstanceLayerProperties()
        layers = [lay.layerName for lay in layers]
        if 'VK_LAYER_KHRONOS_validation' in layers:
            layers = ['VK_LAYER_KHRONOS_validation']
        elif 'VK_LAYER_LUNARG_standard_validation' in layers:
            layers = ['VK_LAYER_LUNARG_standard_validation']
        else:
            layers = []
        extensions = [] # headless rendering

        # Vulkan: instance information
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers
        )

        # Vulkan: instance
        self._instance = vk.vkCreateInstance(create_info, None)

    def _findPhysicalDevice(self):
        # Vulkan: select device
        physical_devices = vk.vkEnumeratePhysicalDevices(self._instance)
        self._physical_device = physical_devices[0]
        for i in range(len(physical_devices) - 1, -1, -1):
            props = vk.vkGetPhysicalDeviceProperties(physical_devices[i])
            if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                self._physical_device = physical_devices[i]
        device_props = vk.vkGetPhysicalDeviceProperties(self._physical_device)
        print(f"Vk Device: {device_props.deviceName}")

    def _checkVideoEncodingSupport(self):
        # Vulkan: test if hardware video encoding is supported
        extensions = vk.vkEnumerateDeviceExtensionProperties(physicalDevice=self._physical_device, pLayerName=None)
        extensions = [ext.extensionName for ext in extensions]
        if 'VK_KHR_video_queue' in extensions and 'VK_KHR_video_encode_queue' in extensions:
            self._gpu_video_encode = True
        if not self._gpu_video_encode:
            print(f"Vk: WARNING> GPU video encoding not supported")

    def _findQueueFamilies(self):
        # Vulkan: find single graphics queue
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self._physical_device)
        for i in range(len(queue_families)):
            if queue_families[i].queueCount > 0 and (queue_families[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT):
                self._queue_family_index = i
        if self._queue_family_index < 0:
            print(f"Vk: WARNING> no graphic queue family found")
            
    def _createLogicalDevice(self):
        # Vulkan: queue information
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self._queue_family_index,
            queueCount=1,
            pQueuePriorities=[1],
            flags=0
        )

        # Vulkan: create logical device
        extensions = []
        if self._gpu_video_encode:
            extensions.append('VK_KHR_video_queue')
            extensions.append('VK_KHR_video_encode_queue')
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=queue_create_info,
            flags=0,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        self._logical_device = vk.vkCreateDevice(self._physical_device, device_create_info, None)

        # Vulkan: get graphics queue
        self._graphic_queue = vk.vkGetDeviceQueue(device=self._logical_device,
                                                  queueFamilyIndex=self._queue_family_index,
                                                  queueIndex=0)

        # Vulkan: get physical device memory properties
        self._device_mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice=self._physical_device)

    def _createSemaphores(self):
        semaphore_info = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            flags=0
        )
        self._semaphore_image_available = vk.vkCreateSemaphore(self._logical_device, semaphore_info, None)
        self._semaphore_render_finished = vk.vkCreateSemaphore(self._logical_device, semaphore_info, None)

    def _createCommandPool(self):
        # Vulkan: command pool
        command_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self._queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self._command_pool = vk.vkCreateCommandPool(self._logical_device, command_pool_info, None)

    def _createVertexBuffer(self):
        # Vulkan: vertex data (X,Y,Z,R,G,B)
        vertex_data = np.array([
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0 ,0.0, 1.0
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 2
        ], dtype=np.uint16)
        
        # Vulkan: command buffer for copy operation
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        copy_command_buffer = vk.vkAllocateCommandBuffers(self._logical_device, command_buffer_info)[0]
        
        # Vulkan: create buffers for vertex and triangle index data (and transfer data)
        self._vertex_object["vertices"] = self._createGpuBuffer(vertex_data,
                                                                vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                copy_command_buffer)
        self._vertex_object["indices"] = self._createGpuBuffer(indices,
                                                               vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                               copy_command_buffer)
        
        # Vulkan: free copy command buffer
        vk.vkFreeCommandBuffers(device=self._logical_device,
                                commandPool=self._command_pool,
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])

        # Vulkan: vertex input binding description (attribute size per vertex) 
        self._vertex_binding_desc = vk.VkVertexInputBindingDescription(
            binding=0,
            stride=24, # X,Y,Z,red,green,blue --> six 4-byte floats = 24 bytes
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX
        )

        # Vulkan: vertex attributes (position and color)
        vertex_position_attrib = vk.VkVertexInputAttributeDescription(
            binding=0, # match binding from vertex_binding_desc
            location=0, # match location in shaders
            format=vk.VK_FORMAT_R32G32B32_SFLOAT, # 3-component vector of floats
            offset=0 # position is 0 bytes from start of vertex data
        )
        vertex_color_attrib = vk.VkVertexInputAttributeDescription(
            binding=0, # match binding from vertex_binding_desc
            location=1, # match location in shaders
            format=vk.VK_FORMAT_R32G32B32_SFLOAT, # 3-component vector of floats
            offset=12 # color is 12 bytes from start of vertex data (X,Y,Z is in front)
        )
        self._vertex_attribs_desc = [vertex_position_attrib, vertex_color_attrib]
        
    def _createUniformBuffer(self):
        # Vulkan: copy uniform data to host accessible buffer memory
        buffer_info = vk.VkBufferCreateInfo(
		    sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		    size=glm.sizeof(glm.mat4), # MVP 4x4 matrix
		    usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        )
        
        self._uniforms["buffer"] = vk.vkCreateBuffer(device=self._logical_device,
                                                 pCreateInfo=buffer_info,
                                                 pAllocator=None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(device=self._logical_device,
                                                    buffer=self._uniforms["buffer"])

        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        self._uniforms["memory"] = vk.vkAllocateMemory(device=self._logical_device,
                                                       pAllocateInfo=mem_alloc_info,
                                                       pAllocator=None)
        vk.vkBindBufferMemory(device=self._logical_device,
                              buffer=self._uniforms["buffer"],
                              memory=self._uniforms["memory"],
                              memoryOffset=0)
        self._updateUniformData()

    def _createImageViews(self):
        # Create color image view 
        self._color_format = vk.VK_FORMAT_R8G8B8A8_UNORM
        color_usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        color_aspect_mask = vk.VK_IMAGE_ASPECT_COLOR_BIT
        
        color_img_view = self._createImageView(self._color_format, color_usage, color_aspect_mask)
        self._color_attachment["image"] = color_img_view["image"]
        self._color_attachment["memory"] = color_img_view["memory"]
        self._color_image_view = color_img_view["image_view"]
        
        # Create depth image view
        self._depth_format = self._getSupportedDepthFormat()
        print(f"Vk Image Depth Format: {self._getDepthFormatName(self._depth_format)}")
        depth_usage = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        depth_aspect_mask = vk.VK_IMAGE_ASPECT_DEPTH_BIT
        if self._depth_format >= vk.VK_FORMAT_D16_UNORM_S8_UINT:
            depth_aspect_mask |= vk.VK_IMAGE_ASPECT_STENCIL_BIT
        
        depth_img_view = self._createImageView(self._depth_format, depth_usage, depth_aspect_mask)
        self._depth_attachment["image"] = depth_img_view["image"]
        self._depth_attachment["memory"] = depth_img_view["memory"]
        self._depth_image_view = depth_img_view["image_view"]

    def _createRenderPass(self):
        # Vulkan: create color and depth attachments
        color_attachement_desc = vk.VkAttachmentDescription(
            format=self._color_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            flags=0
        )
        color_reference = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        depth_attachement_desc = vk.VkAttachmentDescription(
            format=self._depth_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            flags=0
        )
        depth_reference = vk.VkAttachmentReference(
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL 
        )
        
        # Vulkan: create subpass for color and depth
        sub_pass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputAttachmentCount=0,
            pInputAttachments=None,
            colorAttachmentCount=1,
            pColorAttachments=[color_reference],
            pResolveAttachments=None,
            pDepthStencilAttachment=depth_reference,
            preserveAttachmentCount=0,
            pPreserveAttachments=None,
            flags=0
        )
        
        # Vulkan: create subpass dependencies
        dependency1 = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT
        )
        dependency2 = vk.VkSubpassDependency(
            srcSubpass=0,
            dstSubpass=vk.VK_SUBPASS_EXTERNAL,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT,
            dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT
        )
        
        # Vulkan: create render pass
        render_pass_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=2,
            pAttachments=[color_attachement_desc, depth_attachement_desc],
            subpassCount=1,
            pSubpasses=[sub_pass],
            dependencyCount=2,
            pDependencies=[dependency1, dependency2],
            flags=0
        )
        self._render_pass = vk.vkCreateRenderPass(device=self._logical_device,
                                                  pCreateInfo=render_pass_info,
                                                  pAllocator=None)

    def _createFramebuffers(self):
        framebuffer_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self._render_pass,
            attachmentCount=2,
            pAttachments=[self._color_image_view, self._depth_image_view],
            width=self._width,
			height=self._height,
			layers=1,
            flags=0
        )
        self._framebuffer = vk.vkCreateFramebuffer(device=self._logical_device,
                                                   pCreateInfo=framebuffer_info,
                                                   pAllocator=None)

    def _createGraphicsPipeline(self):
        vert_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.vert.spv")
        frag_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.frag.spv")
        print("Vk Create Shader Modules: success")
        
        
    
    def _findMemoryTypeIndex(self, supported_memory_indices, req_properties):
        for i in range(self._device_mem_props.memoryTypeCount):
            supported = supported_memory_indices & (1 << i)
            sufficient = (self._device_mem_props.memoryTypes[i].propertyFlags & req_properties) == req_properties
            if supported and sufficient:
                return i
        return -1

    def _createGpuBuffer(self, data, usage, copy_command_buffer):
        # Vulkan: copy data to host accessible buffer memory
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=data.nbytes,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        )
        staging_buffer = vk.vkCreateBuffer(device=self._logical_device,
                                           pCreateInfo=buffer_info,
                                           pAllocator=None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(device=self._logical_device,
                                                    buffer=staging_buffer)
        
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        staging_memory = vk.vkAllocateMemory(device=self._logical_device,
                                             pAllocateInfo=mem_alloc_info,
                                             pAllocator=None)
        memory_location = vk.vkMapMemory(device=self._logical_device,
                                         memory=staging_memory,
                                         offset=0,
                                         size=data.nbytes,
                                         flags=0)
        ffi.memmove(memory_location, data, data.nbytes)
        vk.vkUnmapMemory(device=self._logical_device, memory=staging_memory)
        vk.vkBindBufferMemory(device=self._logical_device,
                              buffer=staging_buffer,
                              memory=staging_memory,
                              memoryOffset=0)
                              
        # Vulkan: allocate a GPU buffer for data
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=data.nbytes,
            usage=usage | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )
        gpu_buffer = vk.vkCreateBuffer(device=self._logical_device,
                                       pCreateInfo=buffer_info,
                                       pAllocator=None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(device=self._logical_device,
                                                    buffer=gpu_buffer)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        gpu_memory = vk.vkAllocateMemory(device=self._logical_device,
                                         pAllocateInfo=mem_alloc_info,
                                         pAllocator=None)
        vk.vkBindBufferMemory(device=self._logical_device,
                              buffer=gpu_buffer,
                              memory=gpu_memory,
                              memoryOffset=0)

        # Vulkan: copy data from host visible buffer to gpu buffer
        buffer_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=buffer_begin_info)
        copy_region = vk.VkBufferCopy(size=data.nbytes)
        vk.vkCmdCopyBuffer(commandBuffer=copy_command_buffer,
                           srcBuffer=staging_buffer,
                           dstBuffer=gpu_buffer,
                           regionCount=1,
                           pRegions=[copy_region])
        vk.vkEndCommandBuffer(copy_command_buffer)
        
        # Vulkan: submit to queue
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[copy_command_buffer]
        )
        vk.vkQueueSubmit(queue=self._graphic_queue,
                         submitCount=1, 
                         pSubmits=[submit_info],
                         fence=vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(queue=self._graphic_queue)

        # Vulkan: clean up
        vk.vkDestroyBuffer(device=self._logical_device,
                           buffer=staging_buffer,
                           pAllocator=None)
        vk.vkFreeMemory(device=self._logical_device,
                        memory=staging_memory,
                        pAllocator=None)

        # Return created GPU buffer and memory
        return {"buffer": gpu_buffer, "memory": gpu_memory}

    def _updateUniformData(self):
        # TODO: animation -> update model matrix
        mvp_transform = glm.mat4() # projection * model * view
        
        # Vulkan: copy transformation matrix
        memory_location = vk.vkMapMemory(device=self._logical_device,
                                         memory=self._uniforms["memory"],
                                         offset=0,
                                         size=glm.sizeof(glm.mat4),
                                         flags=0)
        ffi.memmove(memory_location, glm.value_ptr(mvp_transform), glm.sizeof(glm.mat4))
        vk.vkUnmapMemory(device=self._logical_device, memory=self._uniforms["memory"])

    def _getSupportedDepthFormat(self):
        # Vulkan: find highest precision packed format that is supported by device
        possible_formats = [
            vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
            vk.VK_FORMAT_D32_SFLOAT,
            vk.VK_FORMAT_D24_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM
        ]
        for fmt in possible_formats:
            format_props = vk.vkGetPhysicalDeviceFormatProperties(physicalDevice=self._physical_device,
                                                                  format=fmt)
            if format_props.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return fmt
        return 0
        
    def _getDepthFormatName(self, depth_format):
        fmt_name = "Unsupported depth format"
        if depth_format == vk.VK_FORMAT_D32_SFLOAT_S8_UINT:
            fmt_name = "VK_FORMAT_D32_SFLOAT_S8_UINT"
        elif depth_format == vk.VK_FORMAT_D32_SFLOAT:
            fmt_name = "VK_FORMAT_D32_SFLOAT"
        elif depth_format == vk.VK_FORMAT_D24_UNORM_S8_UINT:
            fmt_name = "VK_FORMAT_D24_UNORM_S8_UINT"
        elif depth_format == vk.VK_FORMAT_D16_UNORM_S8_UINT:
            fmt_name = "VK_FORMAT_D16_UNORM_S8_UINT"
        elif depth_format == vk.VK_FORMAT_D16_UNORM:
            fmt_name = "VK_FORMAT_D16_UNORM"
        return fmt_name

    def _createImageView(self, img_format, img_usage, aspect_mask):
        # Vulkan: create specified attachment image
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=img_format,
            extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1),
            mipLevels=1,
			arrayLayers=1,
			samples=vk.VK_SAMPLE_COUNT_1_BIT,
			tiling=vk.VK_IMAGE_TILING_OPTIMAL,
			usage=img_usage,
            flags=0
        )
        image = vk.vkCreateImage(device=self._logical_device,
                                 pCreateInfo=image_info,
                                 pAllocator=None)
        mem_reqs = vk.vkGetImageMemoryRequirements(device=self._logical_device, image=image)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        memory = vk.vkAllocateMemory(device=self._logical_device,
                                     pAllocateInfo=mem_alloc_info,
                                     pAllocator=None)
        vk.vkBindImageMemory(device=self._logical_device, image=image, memory=memory, memoryOffset=0)

        # Vulkan: create image view for specified attachment image
        image_view_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			image=image,
			viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
			format=img_format,
			components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY
            ),
			subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=aspect_mask,
			    baseMipLevel=0,
			    levelCount=1,
			    baseArrayLayer=0,
			    layerCount=1
            )
        )
        image_view = vk.vkCreateImageView(device=self._logical_device,
                                          pCreateInfo=image_view_info,
                                          pAllocator=None)

        return {"image": image, "memory": memory, "image_view": image_view}

    def _createShaderModule(self, filename):
        file = open(filename, 'rb')
        shader_src = file.read()
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_src),
            pCode=shader_src,
            flags=0
        )
        
        return vk.vkCreateShaderModule(device=self._logical_device, 
                                       pCreateInfo=shader_module_info,
                                       pAllocator=None)

    """
    def _initMesh(self):
        # Vulkan: vertex input binding description (attribute size per vertex) 
        vertex_binding_desc = vk.VkVertexInputBindingDescription(
            binding=0,
            stride=24, # X,Y,Z,red,green,blue --> six 4-byte floats = 24 bytes
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        
        # Vulkan: vertex attributes (position and color)
        vertex_position_attrib = vk.VkVertexInputAttributeDescription(
            binding=0, # match binding from vertex_binding_desc
            location=0, # match location in shaders
            format=vk.VK_FORMAT_R32G32B32_SFLOAT, # 3-component vector of floats
            offset=0 # position is 0 bytes from start of vertex data
        )
        vertex_color_attrib = vk.VkVertexInputAttributeDescription(
            binding=0, # match binding from vertex_binding_desc
            location=1, # match location in shaders
            format=vk.VK_FORMAT_R32G32B32_SFLOAT, # 3-component vector of floats
            offset=12 # color is 12 bytes from start of vertex data (X,Y,Z is in front)
        )
        vertex_attribs_desc = [vertex_position_attrib, vertex_color_attrib]

        # Vulkan: pipeline vertex input state information
        vertex_input_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[vertex_binding_desc],
            vertexAttributeDescriptionCount=len(vertex_attribs_desc),
            pVertexAttributeDescriptions=vertex_attribs_desc
        )
        
        # Vulkan: vertex data (X,Y,Z,R,G,B)
        vertex_data = np.array([
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0 ,0.0, 1.0
        ], dtype=np.float32)
        
        # Vulkan: vertex buffer
        vertex_buffer_info = vk.VkBufferCreateInfo(
            size=vertex_data.nbytes,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        vertex_buffer = vk.vkCreateBuffer(device=self._logical_device, pCreateInfo=vertex_buffer_info, pAllocator=None)

        # Vulkan: allocate memory for vertex buffer and bind it
        memory_requirements = vk.vkGetBufferMemoryRequirements(device=self._logical_device, buffer=vertex_buffer)
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=memory_requirements.size,
            memoryTypeIndex = self._findMemoryTypeIndex(
                memory_requirements.memoryTypeBits, 
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
        )
        vertex_buffer_mem = vk.vkAllocateMemory(device=self._logical_device, pAllocateInfo=alloc_info, pAllocator=None)
        vk.vkBindBufferMemory(device=self._logical_device, buffer=vertex_buffer, memory=vertex_buffer_mem, memoryOffset=0)
    """

