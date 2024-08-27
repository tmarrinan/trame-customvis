import time
import asyncio
import glm
import numpy as np
import vulkan as vk
from cffi import FFI

ffi = FFI()

# Custom visualization class for drawing a circle
class ExVkTriangle:
    def __init__(self, width, height):
        # Window size
        self._width = width
        self._height = height
        # Triangle position/velocity
        self._triangle_center = [self._width // 2, self._height // 2]
        self._velocity_x = 100
        self._velocity_y = 60
        # Trame image options
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}
        self._gpu_video_encode = False
        # Animation timing
        self._start_time = round(time.time_ns() / 1000000)
        self._prev_time = self._start_time
        # Vulkan variables
        self._vk = {}
        
        # Initialize
        self._initVulkan()

    def renderFrame(self):
        # Animate
        now = round(time.time_ns() / 1000000)
        dt = (now - self._prev_time) / 1000
        
        dx = round(self._velocity_x * dt)
        dy = round(self._velocity_y * dt)
        if self._triangle_center[0] + dx < 0:
            self._triangle_center[0] = 0
            self._velocity_x *= -1
        elif self._triangle_center[0] + dx > self._width:
            self._triangle_center[0] = self._width
            self._velocity_x *= -1
        else:
            self._triangle_center[0] += dx
        if self._triangle_center[1] + dy < 0:
            self._triangle_center[1] = 0
            self._velocity_y *= -1
        elif self._triangle_center[1] + dy > self._height:
            self._triangle_center[1] = self._height
            self._triangle_center *= -1
        else:
            self._triangle_center[1] += dy
        
        # Update model matrix
        #proj = glm.ortho(0, self._width, 0, self._height)
        #view = glm.mat4(1)
        #model = glm.scale(glm.mat4(1), glm.vec3(100, 100, 1))
        #model = glm.translate(model, glm.vec3(self._triangle_center[0], self._triangle_center[1], 0))
        #self._mvp_transform = proj * view * model
        
        # Vulkan: submit render command to graphics queue
        self._submitWork(self._vk["graphic_cmd_buffer"], self._vk["graphic_queue"])
        vk.vkDeviceWaitIdle(device=self._vk["device"])

        # Update render time
        self._prev_time = now

    def _initVulkan(self):
        self._vk["instance"] = self._createInstance()
        self._vk["physical_device"] = self._findPhysicalDevice()
        self._vk["graphic_family_index"] = self._getGraphicQueueFamilyIndex()
        self._vk["device"] = self._createLogicalDevice()
        self._vk["graphic_queue"] = self._getGraphicQueue()
        self._vk["render_pass"] = self._createRenderPass()
        self._vk.update(self._createGraphicsPipeline())
        self._vk.update(self._createColorAttachment())
        self._vk["framebuffer"] = self._createFramebuffer()
        self._vk["command_pool"] = self._createCommandPool()
        self._vk["graphic_cmd_buffer"] = self._createCommandBuffer()
        self._vk.update(self._createTriangleMesh())

        print("Vulkan: application objects")
        for key,item in self._vk.items():
            print(f"  {key:24s}: {item}")

        self._recordDrawCommands()

    def _createInstance(self):
        # Get Vulkan API version
        version = vk.vkEnumerateInstanceVersion()
        version_str = f"{vk.VK_VERSION_MAJOR(version)}.{vk.VK_VERSION_MINOR(version)}.{vk.VK_VERSION_PATCH(version)}"
        print(f"Vulkan: found support for version {version_str}")
        
        # Set the patch to 0 for best compatibility/stability
        version &= ~(0xFFF)
        version_str = f"{vk.VK_VERSION_MAJOR(version)}.{vk.VK_VERSION_MINOR(version)}.{vk.VK_VERSION_PATCH(version)}"
        print(f"Vulkan: creating instance using version {version_str}")
        
        # Create application information
        app_info = vk.VkApplicationInfo(
            pApplicationName = "Vulkan Headless Triangle",
            applicationVersion = version,
            pEngineName = "No engine",
            engineVersion = version,
            apiVersion = version
        )
        
        # Select instance layers and extensions
        layers = vk.vkEnumerateInstanceLayerProperties()
        layers = [lay.layerName for lay in layers]
        extensions = [] # headless rendering
        if "VK_LAYER_KHRONOS_validation" in layers:
            layers = ["VK_LAYER_KHRONOS_validation"]
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        elif "VK_LAYER_LUNARG_standard_validation" in layers:
            layers = ["VK_LAYER_LUNARG_standard_validation"]
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        else:
            layers = []
        
        # Create instance information
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        # Create instance
        instance = vk.vkCreateInstance(pCreateInfo=create_info, pAllocator=None)
        
        # Set up debug callback
        if vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME in extensions:
            debug_utils_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
                sType=vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                                vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                pfnUserCallback=self._debugMessageCallback,
                pUserData=None,
                flags=0
            )
            vkCreateDebugUtilsMessengerEXT = vk.vkGetInstanceProcAddr(instance=instance, pName="vkCreateDebugUtilsMessengerEXT")
            debug_messenger = vkCreateDebugUtilsMessengerEXT(instance=instance,
                                                             pCreateInfo=debug_utils_info,
                                                             pAllocator=None)

        return instance

    def _findPhysicalDevice(self):
        # Search available devices for discrete GPU
        physical_devices = vk.vkEnumeratePhysicalDevices(instance=self._vk["instance"])
        physical_device = physical_devices[0]
        for i in range(len(physical_devices) - 1, -1, -1):
            props = vk.vkGetPhysicalDeviceProperties(physical_devices[i])
            if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                physical_device = physical_devices[i]
        device_props = vk.vkGetPhysicalDeviceProperties(physical_device)
        print(f"Vulkan: using device {device_props.deviceName}")

        self._checkVideoEncodingSupport(physical_device)

        return physical_device

    def _checkVideoEncodingSupport(self, physical_device):
        # Check if hardware video encoding is supported
        extensions = vk.vkEnumerateDeviceExtensionProperties(physicalDevice=physical_device, pLayerName=None)
        extensions = [ext.extensionName for ext in extensions]
        if 'VK_KHR_video_queue' in extensions and 'VK_KHR_video_encode_queue' in extensions:
            self._gpu_video_encode = True
        if not self._gpu_video_encode:
            print("Vk: WARNING> GPU video encoding not supported")

    def _getGraphicQueueFamilyIndex(self):
        # Find index for a graphics queue
        queue_family_index = -1
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self._vk["physical_device"])
        for i in range(len(queue_families)):
            if queue_families[i].queueCount > 0 and (queue_families[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT):
                queue_family_index = i
        if queue_family_index < 0:
            print(f"Vulkan: WARNING> no graphic queue family found")

        return queue_family_index

    def _createLogicalDevice(self):
        # Create queue information
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self._vk["graphic_family_index"],
            queueCount=1,
            pQueuePriorities=[1.0]
        )

        # Device features must be requested before the device is abstracted
        device_features = vk.VkPhysicalDeviceFeatures()

        # Set up extensions
        extensions = []
        if self._gpu_video_encode:
            extensions.append('VK_KHR_video_queue')
            extensions.append('VK_KHR_video_encode_queue')

        # Create device information
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            pEnabledFeatures = [device_features]
        )

        # Create device
        logical_device = vk.vkCreateDevice(physicalDevice=self._vk["physical_device"],
                                           pCreateInfo=device_create_info,
                                           pAllocator=None)

        return logical_device

    def _getGraphicQueue(self):
        # Get graphics queue
        return vk.vkGetDeviceQueue(device=self._vk["device"],
                                   queueFamilyIndex=self._vk["graphic_family_index"],
                                   queueIndex=0)

    def _createRenderPass(self):
        # Create color attachment description
        color_attachment_desc = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL 
        )

        # Create color attachment reference
        color_attachment_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        # Create subpass description
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=color_attachment_ref
        )

        # Create render pass information
        render_pass_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=color_attachment_desc,
            subpassCount=1,
            pSubpasses=subpass
        )

        return vk.vkCreateRenderPass(device=self._vk["device"],
                                     pCreateInfo=render_pass_info,
                                     pAllocator=None)

    def _createGraphicsPipeline(self):
        # Each vertex is 20 bytes (X,Y,R,G,B)
        triangle_binding_desc = vk.VkVertexInputBindingDescription(binding=0,
                                                                   stride=20,
                                                                   inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX)
        triangle_attrib_desc = [
            # Vertex position attribute (2D)
            vk.VkVertexInputAttributeDescription(binding=0,
                                                 location=0,
                                                 format=vk.VK_FORMAT_R32G32_SFLOAT,
                                                 offset=0),
            # Vertex color attribute (RGB)
            vk.VkVertexInputAttributeDescription(binding=0,
                                                 location=1,
                                                 format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                                                 offset=8)
        ]
        
        # Create vertex input state information
        vertex_input_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions= [triangle_binding_desc],
            vertexAttributeDescriptionCount=2,
            pVertexAttributeDescriptions=triangle_attrib_desc
        )
        
        # Load in vertex and fragment shaders
        vert_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.vert.spv")
        frag_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.frag.spv")
        
        # Vulkan: set up shader stage information
        vertex_shader_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            module=vert_shader_module,
            pName="main"
        )
        fragment_shader_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=frag_shader_module,
            pName="main"
        )
        
        # Create input assembly information
        input_assembly_info = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE
        )
        
        # Set up viewport and scissor
        viewport = vk.VkViewport(x=0,
                                 y=0,
                                 width=self._width,
                                 height=self._height,
                                 minDepth=0.0,
                                 maxDepth=1.0)
        scissor = vk.VkRect2D(offset=[0,0],                       # vk.VkOffset2D(x=0, y=0),
                              extent=[self._width, self._height]) # vk.VkExtent2D(width=self._width, height=self._height)

        # Create viewport state information
        viewport_state_info = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor
        )

        # Create rasterization state information
        raterization_state_info = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE
        )
        
        # Create multisampling state information
        multisampling_state_info = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
        )
        
        # Create color blending state information
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT |
                                                                                       vk.VK_COLOR_COMPONENT_G_BIT |
                                                                                       vk.VK_COLOR_COMPONENT_B_BIT |
                                                                                       vk.VK_COLOR_COMPONENT_A_BIT,
                                                                        blendEnable=vk.VK_FALSE)
        color_blend_info = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1,
            pAttachments=color_blend_attachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )
        
        # Set up push constant
        push_constant_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                                                     offset=0,
                                                     size=64)

        # Create pipeline layout information
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range],
            setLayoutCount=0
        )

        # Create pipeline layout
        pipeline_layout = vk.vkCreatePipelineLayout(device=self._vk["device"],
                                                    pCreateInfo=pipeline_layout_info,
                                                    pAllocator=None)

        # Create graphic pipeline information
        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2,
            pStages=[vertex_shader_info, fragment_shader_info],
            pVertexInputState=vertex_input_info,
            pInputAssemblyState=input_assembly_info,
            pViewportState=viewport_state_info,
            pRasterizationState=raterization_state_info,
            pMultisampleState=multisampling_state_info,
            pDepthStencilState=None,
            pColorBlendState=color_blend_info,
            layout=pipeline_layout,
            renderPass=self._vk["render_pass"],
            subpass=0
        )
        
        # Create graphics pipeline
        graphics_pipeline = vk.vkCreateGraphicsPipelines(device=self._vk["device"],
                                                         pipelineCache=vk.VK_NULL_HANDLE,
                                                         createInfoCount=1,
                                                         pCreateInfos=pipeline_info,
                                                         pAllocator=None)[0]

        # Vulkan: free shader modules
        vk.vkDestroyShaderModule(device=self._vk["device"], shaderModule=vert_shader_module, pAllocator=None)
        vk.vkDestroyShaderModule(device=self._vk["device"], shaderModule=frag_shader_module, pAllocator=None)

        return {"pipeline_layout": pipeline_layout, "pipeline": graphics_pipeline}

    def _createColorAttachment(self):
        # Create image view for framebuffer
        color_attachment = self._createImageView(vk.VK_FORMAT_R8G8B8A8_UNORM,
                                                 vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                 vk.VK_IMAGE_ASPECT_COLOR_BIT)

        return {"color_attachment_image": color_attachment["image"], "color_attachment_memory": color_attachment["memory"],
                "color_attachment_view": color_attachment["view"]}

    def _createFramebuffer(self):
        # Create framebuffer information
        framebuffer_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self._vk["render_pass"],
            attachmentCount=1,
            pAttachments=[self._vk["color_attachment_view"]],
            width=self._width,
            height=self._height,
            layers=1
        )

        # Create framebuffer
        return vk.vkCreateFramebuffer(device=self._vk["device"], pCreateInfo=framebuffer_info, pAllocator=None)

    def _createCommandPool(self):
        # Create command pool information
        command_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self._vk["graphic_family_index"],
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        
        # Create command pool
        return vk.vkCreateCommandPool(device=self._vk["device"], pCreateInfo=command_pool_info, pAllocator=None)

    def _createCommandBuffer(self):
        # Command buffer allocation information
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._vk["command_pool"],
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        # Allocate command buffer
        return vk.vkAllocateCommandBuffers(device=self._vk["device"], pAllocateInfo=command_buffer_info)[0]

    def _createTriangleMesh(self):
        # Create vertex data (X,Y,R,G,B)
        vertices = np.array([ 0.0, -0.05, 1.0, 0.0, 0.0,
                              0.05, 0.05, 0.0, 1.0, 0.0,
                             -0.05, 0.05, 0.0, 0.0, 1.0],
                            dtype=np.float32)

        # Create vertex buffer
        vertex_buffer = self._createBuffer(vertices.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Copy vertex data to buffer
        memory_location = vk.vkMapMemory(device=self._vk["device"], memory=vertex_buffer["memory"],
                                         offset=0, size=vertices.nbytes, flags=0)
        ffi.memmove(memory_location, vertices, vertices.nbytes)
        #TEST
        #memory_array = np.frombuffer(memory_location, dtype=vertices.dtype, count=vertices.size, offset=0)
        #print(memory_array)
            
        vk.vkUnmapMemory(device=self._vk["device"], memory=vertex_buffer["memory"])
        
        return {"vbo_vert_buffer": vertex_buffer["buffer"], "vbo_vert_memory": vertex_buffer["memory"]}

    def _recordDrawCommands(self):
        # Prepare data for recording command buffers
        command_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        
        # Record command buffer
        vk.vkBeginCommandBuffer(commandBuffer=self._vk["graphic_cmd_buffer"], pBeginInfo=command_begin_info)
        
        # Set clear color value
        clear_color = vk.VkClearValue(
			color=vk.VkClearColorValue(float32=[0.4, 0.1, 0.6, 1.0]) # R,G,B,A
        )
        
        # Render pass begin information
        render_pass_begin_info = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderArea=[[0, 0], [self._width, self._height]], #vk.VkRect2D(extent=vk.VkExtent2D(width=self._width, height=self._height)),
            clearValueCount=1,
            pClearValues=[clear_color],
            renderPass=self._vk["render_pass"],
            framebuffer=self._vk["framebuffer"]
        )
        
        # Begin render pass
        vk.vkCmdBeginRenderPass(commandBuffer=self._vk["graphic_cmd_buffer"],
                                pRenderPassBegin=render_pass_begin_info,
                                contents=vk.VK_SUBPASS_CONTENTS_INLINE)

        # Bind pipeline
        vk.vkCmdBindPipeline(commandBuffer=self._vk["graphic_cmd_buffer"],
                             pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline=self._vk["pipeline"])

        # Bind vertex buffer object
        vk.vkCmdBindVertexBuffers(commandBuffer=self._vk["graphic_cmd_buffer"],
                                  firstBinding=0,
                                  bindingCount=1,
                                  pBuffers=[self._vk["vbo_vert_buffer"]],
                                  pOffsets=[0])
        
        # MVP push constant
        triangle_positions = []
        for x in [-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75]:
            for y in [-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75]:
                triangle_positions.append(glm.vec3(x, y, 0))
        for pos in triangle_positions:
            mvp_mat4 = glm.translate(glm.mat4(1.0), pos)
            mvp_raw = ffi.cast("float *", ffi.from_buffer(mvp_mat4))
            vk.vkCmdPushConstants(commandBuffer=self._vk["graphic_cmd_buffer"],
                                  layout=self._vk["pipeline_layout"],
                                  stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                                  offset=0,
                                  size=glm.sizeof(glm.mat4),
                                  pValues=mvp_raw)
            vk.vkCmdDraw(commandBuffer=self._vk["graphic_cmd_buffer"], vertexCount=3, 
                         instanceCount=1, firstVertex=0, firstInstance=0)

        # End render pass and command buffer
        vk.vkCmdEndRenderPass(commandBuffer=self._vk["graphic_cmd_buffer"])
        vk.vkEndCommandBuffer(commandBuffer=self._vk["graphic_cmd_buffer"])

    def _createShaderModule(self, filename):
        file = open(filename, 'rb')
        shader_src = file.read()
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_src),
            pCode=shader_src
        )
        
        return vk.vkCreateShaderModule(device=self._vk["device"], 
                                       pCreateInfo=shader_module_info,
                                       pAllocator=None)

    def _findMemoryTypeIndex(self, supported_memory_indices, req_properties):
        device_mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice=self._vk["physical_device"])
        for i in range(device_mem_props.memoryTypeCount):
            supported = supported_memory_indices & (1 << i)
            sufficient = (device_mem_props.memoryTypes[i].propertyFlags & req_properties) == req_properties
            if supported and sufficient:
                return i
        return -1

    def _createImageView(self, img_format, img_usage, aspect_mask):
        # Create specified attachment image
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=img_format,
            extent=[self._width, self._height, 1], # vk.VkExtent3D(width=self._width, height=self._height, depth=1),
            mipLevels=1,
			arrayLayers=1,
			samples=vk.VK_SAMPLE_COUNT_1_BIT,
			tiling=vk.VK_IMAGE_TILING_OPTIMAL,
			usage=img_usage,
            flags=0
        )
        image = vk.vkCreateImage(device=self._vk["device"],
                                 pCreateInfo=image_info,
                                 pAllocator=None)
        mem_reqs = vk.vkGetImageMemoryRequirements(device=self._vk["device"], image=image)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        memory = vk.vkAllocateMemory(device=self._vk["device"], pAllocateInfo=mem_alloc_info, pAllocator=None)
        vk.vkBindImageMemory(device=self._vk["device"], image=image, memory=memory, memoryOffset=0)

        # Create image view for specified attachment image
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
            ),
            flags=0
        )
        image_view = vk.vkCreateImageView(device=self._vk["device"],
                                          pCreateInfo=image_view_info,
                                          pAllocator=None)

        return {"image": image, "memory": memory, "view": image_view}

    def _createBuffer(self, size, usage_flags, memory_prop_flags):
        # Create buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage_flags,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        buffer = vk.vkCreateBuffer(device=self._vk["device"], pCreateInfo=buffer_info, pAllocator=None)
        
        # Get buffer memory requirments
        mem_reqs = vk.vkGetBufferMemoryRequirements(device=self._vk["device"], buffer=buffer)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, memory_prop_flags)
        
        # Allocate memory
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        memory = vk.vkAllocateMemory(device=self._vk["device"], pAllocateInfo=mem_alloc_info, pAllocator=None)

        # Bind buffer memory
        vk.vkBindBufferMemory(device=self._vk["device"], buffer=buffer, memory=memory, memoryOffset=0)
        
        return {"buffer": buffer, "memory": memory}

    def _submitWork(self, command_buffer, queue):
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )

        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=0
        )
        fence = vk.vkCreateFence(device=self._vk["device"],
                                 pCreateInfo=fence_info,
                                 pAllocator=None)

        vk.vkQueueSubmit(queue=queue,
                         submitCount=1, 
                         pSubmits=[submit_info],
                         fence=fence)
        vk.vkWaitForFences(device=self._vk["device"],
                           fenceCount=1,
                           pFences=[fence],
                           waitAll=vk.VK_TRUE,
                           timeout=np.iinfo(np.uint64).max)
        vk.vkDestroyFence(device=self._vk["device"],
                          fence=fence,
                          pAllocator=None)

    def _insertImageMemoryBarrier(self, command_buffer, image, src_access_mask, dst_access_mask,
                                  old_img_layout, new_img_layout, src_stage_mask, dst_stage_mask,
                                  subresource_range):
        memory_barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            srcAccessMask=src_access_mask,
            dstAccessMask=dst_access_mask,
            oldLayout=old_img_layout,
            newLayout=new_img_layout,
            image=image,
            subresourceRange=subresource_range
        )
        vk.vkCmdPipelineBarrier(commandBuffer=command_buffer,
				                srcStageMask=src_stage_mask,
				                dstStageMask=dst_stage_mask,
				                dependencyFlags=0,
				                memoryBarrierCount=0,
                                pMemoryBarriers=[],
				                bufferMemoryBarrierCount=0,
                                pBufferMemoryBarriers=[],
                                imageMemoryBarrierCount=1,
                                pImageMemoryBarriers=[memory_barrier])

    def _getRawImage(self):
        # Vulkan: create destination image to copy to and read the memory from
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
			format=vk.VK_FORMAT_R8G8B8A8_UNORM,
			extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1),
			arrayLayers=1,
			mipLevels=1,
			initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
			samples=vk.VK_SAMPLE_COUNT_1_BIT,
			tiling=vk.VK_IMAGE_TILING_LINEAR,
			usage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            flags=0
        )
        image = vk.vkCreateImage(device=self._vk["device"], pCreateInfo=image_info, pAllocator=None)
        
        # Vulkan: create and bind memory for the image
        mem_reqs = vk.vkGetImageMemoryRequirements(device=self._vk["device"], image=image)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits,
                                                   vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        image_memory = vk.vkAllocateMemory(device=self._vk["device"], pAllocateInfo=mem_alloc_info, pAllocator=None)
        vk.vkBindImageMemory(device=self._vk["device"], image=image, memory=image_memory, memoryOffset=0)

        # Vulkan: copy rendered image to host visible destination image
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._vk["command_pool"],
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        copy_command_buffer = vk.vkAllocateCommandBuffers(self._vk["device"], command_buffer_info)[0]
        
        buffer_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=0
        )
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=buffer_begin_info)

        subresource_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        self._insertImageMemoryBarrier(copy_command_buffer, image, 0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                       vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       subresource_range)
        image_copy_region = vk.VkImageCopy(
            srcSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            dstSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1)
        )
        vk.vkCmdCopyImage(commandBuffer=copy_command_buffer,
                          srcImage=self._vk["color_attachment_image"],
				          srcImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dstImage=image,
                          dstImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          regionCount=1,
                          pRegions=[image_copy_region])
        self._insertImageMemoryBarrier(copy_command_buffer, image, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                       vk.VK_ACCESS_MEMORY_READ_BIT, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       vk.VK_PIPELINE_STAGE_TRANSFER_BIT, subresource_range)

        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)
        
        self._submitWork(copy_command_buffer, self._vk["graphic_queue"])
        
        # Vulkan: get layout of the image (including row pitch)
        subresource = vk.VkImageSubresource(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT)
        subresource_layout = vk.vkGetImageSubresourceLayout(device=self._vk["device"],
                                                            image=image,
                                                            pSubresource=[subresource])
        
        # Vulkan: map image memory so we can read it
        memory_location = vk.vkMapMemory(device=self._vk["device"],
                                         memory=image_memory,
                                         offset=0,
                                         size=self._width * self._height * 4,
                                         flags=0)
        
        # Convert memory to numpy array and extract RGB data
        memory_array = np.frombuffer(memory_location, dtype=np.uint8, count=self._width * self._height * 4,
                                     offset=subresource_layout.offset)
        print(f"FIRST PX: {memory_array[0]} {memory_array[1]} {memory_array[2]} {memory_array[3]}")
        for px in range(0, memory_array.size, 4):
            if memory_array[px+0] != 102 or memory_array[px+1] != 25 or memory_array[px+2] != 153:
                print(f"FOUND non-purple px: {px} ({memory_array[px+0]}, {memory_array[px+1]}, {memory_array[px+3]})")
                break
        rgb_img = memory_array[np.mod(np.arange(memory_array.size), 4) != 3]
        
        # TEST -> save PPM image
        file = open("vk_image.ppm", "wb")
        file.write(f"P6\n{self._width} {self._height}\n255\n".encode("utf-8"))
        file.write(rgb_img.tobytes())
        file.close()

        return rgb_img

    def _messageSeverityToString(self, message_severity):
        severity_str = "UNKNOWN"
        if message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            severity_str = "VERBOSE"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            severity_str = "INFO"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity_str = "WARNING"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity_str = "ERROR"
        return severity_str

    def _debugMessageCallback(self, message_severity, message_types, callback_data, user_data):
        message = ffi.string(callback_data.pMessage).decode('utf-8')
        severity = self._messageSeverityToString(message_severity)
        print(f"[VULKAN {severity}]: {message}")
        return vk.VK_FALSE
"""
# Custom visualization class for drawing a circle
class ExVkTriangle:
    def __init__(self, width, height):
        # Window size
        self._width = width
        self._height = height
        # Triangle position/velocity
        self._triangle_center = [self._width // 2, self._height // 2]
        self._velocity_x = 100
        self._velocity_y = 60
        # Trame image options
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}
        self._gpu_video_encode = False
        # Animation timing
        self._start_time = round(time.time_ns() / 1000000)
        self._prev_time = self._start_time
        # Rendering MVP matrix
        self._mvp_transform = glm.mat4(1.0)
        # Vulkan variables
        self._logical_device = None
        self._device_mem_props = None
        self._graphic_queue = None
        self._command_pool = None
        self._color_attachment = None
        self._render_cmd_buffer = None

        # initialie
        self._initVulkan()
        
        # TEST -> render
        self.renderFrame()
        img = self.getFrame()

    def getSize(self):
        return (self._width, self._height)

    def setSize(self, width, height):
        # TODO: resize Vulkan framebuffer
        pass

    def getImageType(self):
        return self._image_type

    def setImageType(self, itype, options={}):
        self._image_type = itype
        if self._image_type == "rgb":
            pass # do nothing
        elif self._image_type == "jpeg":
            self._jpeg_quality = options.get("quality", 92)
        elif self._image_type == "h264":
            self._video_options = options

    def getFrame(self):
        time.sleep(0.5)
        if self._image_type == "rgb":
            return self._getRawImage()
        elif self._image_type == "jpeg":
            return self._getJpegImage()
        elif self._image_type == "h264":
            return self._getH264VideoFrame()
        else:
            return None

    def getRenderTime(self):
        return self._prev_time

    def renderFrame(self):
        # Animate
        now = round(time.time_ns() / 1000000)
        dt = (now - self._prev_time) / 1000
        
        dx = round(self._velocity_x * dt)
        dy = round(self._velocity_y * dt)
        if self._triangle_center[0] + dx < 0:
            self._triangle_center[0] = 0
            self._velocity_x *= -1
        elif self._triangle_center[0] + dx > self._width:
            self._triangle_center[0] = self._width
            self._velocity_x *= -1
        else:
            self._triangle_center[0] += dx
        if self._triangle_center[1] + dy < 0:
            self._triangle_center[1] = 0
            self._velocity_y *= -1
        elif self._triangle_center[1] + dy > self._height:
            self._triangle_center[1] = self._height
            self._triangle_center *= -1
        else:
            self._triangle_center[1] += dy
        
        # Update model matrix
        proj = glm.ortho(0, self._width, 0, self._height)
        view = glm.mat4(1)
        model = glm.scale(glm.mat4(1), glm.vec3(100, 100, 1))
        model = glm.translate(model, glm.vec3(self._triangle_center[0], self._triangle_center[1], 0))
        self._mvp_transform = proj * view * model
        
        # Vulkan: submit render command to graphics queue
        self._submitWork(self._render_cmd_buffer, self._graphic_queue)
        vk.vkDeviceWaitIdle(device=self._logical_device)

        # Update render time
        self._prev_time = now

    def _initVulkan(self):
        # Create application instance
        instance = self._createInstance()
        print(f"instance: {instance}")
        
        # Find physical deivce
        physical_device = self._findPhysicalDevice(instance)
        print(f"physical_device: {physical_device}")
        
        # Check if device supports hardware video encoding
        self._checkVideoEncodingSupport(physical_device)
        
        # Find index for graphic queue family
        queue_family_index = self._findGraphicQueueFamily(physical_device)
        print(f"queue_family_index: {queue_family_index}")
        
        # Create logical device
        self._createLogicalDevice(physical_device, queue_family_index)
        print(f"logical_device: {self._logical_device}")
        
        # Create command pool
        self._createCommandPool(queue_family_index)
        print(f"command_pool: {self._command_pool}")
        
        # Create vertex and index buffers for triangle mesh
        vertex_object = self._createTriangeMesh()
        print(f"vertex_object: {vertex_object}")
        
        # Set color and depth formats
        color_format = vk.VK_FORMAT_R8G8B8A8_UNORM
        depth_format = self._getSupportedDepthFormat(physical_device)
        
        # Create color and depth image views
        framebuffer_attachments = self._createImageViews(color_format, depth_format)
        self._color_attachment = framebuffer_attachments["color"]
        print(f"framebuffer_attachments: {framebuffer_attachments}")
        
        # Create render pass
        render_pass = self._createRenderPass(color_format, depth_format)
        print(f"render_pass: {render_pass}")
        
        # Create framebuffer
        framebuffer = self._createFramebuffer(render_pass,
                                              framebuffer_attachments["color"]["view"],
                                              framebuffer_attachments["depth"]["view"])
        print(f"framebuffer: {framebuffer}")

        # Create graphics pipeline
        graphics_pipeline = self._createGraphicsPipeline(render_pass)
        print(f"graphics_pipeline: {graphics_pipeline}")

        # Create rendering command buffer
        self._createGraphicCommandBuffer(render_pass, framebuffer, graphics_pipeline, vertex_object)
        print(f"render_cmd_buffer: {self._render_cmd_buffer}")
    
    def _createInstance(self):
        # Vulkan: application information
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Vulkan Headless Triangle",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(0, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0)
        )

        # Vulkan: instance layers and extensions
        layers = vk.vkEnumerateInstanceLayerProperties()
        layers = [lay.layerName for lay in layers]
        extensions = [] # headless rendering
        if 'VK_LAYER_KHRONOS_validation' in layers:
            layers = ['VK_LAYER_KHRONOS_validation']
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        elif 'VK_LAYER_LUNARG_standard_validation' in layers:
            layers = ['VK_LAYER_LUNARG_standard_validation']
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        else:
            layers = []
        
        # Vulkan: instance information
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            flags=0
        )

        # Vulkan: instance
        instance = vk.vkCreateInstance(pCreateInfo=create_info, pAllocator=None)
        
        # Vulkan: debug callback
        if vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME in extensions:
            debug_utils_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
                sType=vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                                vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                pfnUserCallback=self._debugMessageCallback,
                pUserData=None,
                flags=0
            )
            vkCreateDebugUtilsMessengerEXT = vk.vkGetInstanceProcAddr(instance=instance, pName="vkCreateDebugUtilsMessengerEXT")
            debug_messenger = vkCreateDebugUtilsMessengerEXT(instance=instance,
                                                             pCreateInfo=debug_utils_info,
                                                             pAllocator=None)
            
            # TEST DEBUG MESSAGE
            #vkDebugReportMessageEXT = vk.vkGetInstanceProcAddr(instance=instance, pName="vkDebugReportMessageEXT")
            #vkDebugReportMessageEXT(instance, vk.VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
            #                        vk.VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, vk.VK_NULL_HANDLE,
            #                        0, 0, "User Debug", "testing debug callback")

        return instance

    def _findPhysicalDevice(self, instance):
        # Vulkan: select device
        physical_devices = vk.vkEnumeratePhysicalDevices(instance=instance)
        physical_device = physical_devices[0]
        for i in range(len(physical_devices) - 1, -1, -1):
            props = vk.vkGetPhysicalDeviceProperties(physical_devices[i])
            if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                physical_device = physical_devices[i]
        device_props = vk.vkGetPhysicalDeviceProperties(physical_device)
        print(f"Vk Device: {device_props.deviceName}")
        return physical_device

    def _checkVideoEncodingSupport(self, physical_device):
        # Vulkan: test if hardware video encoding is supported
        extensions = vk.vkEnumerateDeviceExtensionProperties(physicalDevice=physical_device, pLayerName=None)
        extensions = [ext.extensionName for ext in extensions]
        if 'VK_KHR_video_queue' in extensions and 'VK_KHR_video_encode_queue' in extensions:
            self._gpu_video_encode = True
        if not self._gpu_video_encode:
            print("Vk: WARNING> GPU video encoding not supported")

    def _findGraphicQueueFamily(self, physical_device):
        # Vulkan: find single graphics queue
        queue_family_index = -1
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=physical_device)
        for i in range(len(queue_families)):
            if queue_families[i].queueCount > 0 and (queue_families[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT):
                queue_family_index = i
        if queue_family_index < 0:
            print(f"Vk: WARNING> no graphic queue family found")
        return queue_family_index
            
    def _createLogicalDevice(self, physical_device, queue_family_index):
        # Vulkan: queue information
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[0],
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
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            flags=0
        )
        self._logical_device = vk.vkCreateDevice(physicalDevice=physical_device,
                                                 pCreateInfo=device_create_info,
                                                 pAllocator=None)

        # Vulkan: get graphics queue
        self._graphic_queue = vk.vkGetDeviceQueue(device=self._logical_device,
                                                  queueFamilyIndex=queue_family_index,
                                                  queueIndex=0)

        # Vulkan: get physical device memory properties
        self._device_mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice=physical_device)

    def _createCommandPool(self, queue_family_index):
        # Vulkan: command pool
        command_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self._command_pool = vk.vkCreateCommandPool(device=self._logical_device,
                                                    pCreateInfo=command_pool_info,
                                                    pAllocator=None)

    def _createTriangeMesh(self):
        vertex_object = {"vertices": None, "indices": None}
    
        # Vulkan: vertex data (X,Y,Z,R,G,B)
        vertex_data = np.array([-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                                 0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                                 0.0,  0.5, 0.0, 0.0 ,0.0, 1.0],
                               dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        
        # Vulkan: command buffer for copy operation
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        copy_command_buffer = vk.vkAllocateCommandBuffers(self._logical_device, command_buffer_info)[0]
        
        # TRIANGLE VERTEX DATA
        # Vulkan: create buffers for triangle vertex data
        staging = self._createBuffer(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     vertex_data.nbytes,
                                     vertex_data)

        vertex_object["vertices"] = self._createBuffer(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                       vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                       vertex_data.nbytes)

        # Vulkan: transfer vertex data to GPU buffer
        copy_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=0
        )
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=copy_begin_info)
        copy_region = vk.VkBufferCopy(size=vertex_data.nbytes)
        vk.vkCmdCopyBuffer(commandBuffer=copy_command_buffer,
                           srcBuffer=staging["buffer"],
                           dstBuffer=vertex_object["vertices"]["buffer"],
                           regionCount=1,
                           pRegions=[copy_region])
        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)
        
        self._submitWork(copy_command_buffer, self._graphic_queue)
        
        # Vulkan: free stagin buffer
        vk.vkDestroyBuffer(device=self._logical_device, buffer=staging["buffer"], pAllocator=None)
        vk.vkFreeMemory(device=self._logical_device, memory=staging["memory"], pAllocator=None)
        
        # TRIANGLE FACE INDEX DATA
        # Vulkan: create buffers for triangle face index data
        staging = self._createBuffer(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     indices.nbytes,
                                     indices)

        vertex_object["indices"] = self._createBuffer(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                      vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                      indices.nbytes)

        # Vulkan: transfer face index data to GPU buffer
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=copy_begin_info)
        copy_region = vk.VkBufferCopy(size=indices.nbytes)
        vk.vkCmdCopyBuffer(commandBuffer=copy_command_buffer,
                           srcBuffer=staging["buffer"],
                           dstBuffer=vertex_object["indices"]["buffer"],
                           regionCount=1,
                           pRegions=[copy_region])
        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)
        
        self._submitWork(copy_command_buffer, self._graphic_queue)
        
        # Vulkan: free stagin buffer
        vk.vkDestroyBuffer(device=self._logical_device, buffer=staging["buffer"], pAllocator=None)
        vk.vkFreeMemory(device=self._logical_device, memory=staging["memory"], pAllocator=None)
        
        # Vulkan: free copy command buffer
        vk.vkFreeCommandBuffers(device=self._logical_device,
                                commandPool=self._command_pool,
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])

        return vertex_object

    def _createImageViews(self, color_format, depth_format):
        attachments = {"color": None, "depth": None}
        
        # Create color image view
        color_usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        color_aspect_mask = vk.VK_IMAGE_ASPECT_COLOR_BIT
        
        attachments["color"] = self._createImageView(color_format, color_usage, color_aspect_mask)
        
        # Create depth image view
        print(f"Vk Image Depth Format: {self._getDepthFormatName(depth_format)}")
        depth_usage = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        depth_aspect_mask = vk.VK_IMAGE_ASPECT_DEPTH_BIT
        if depth_format >= vk.VK_FORMAT_D16_UNORM_S8_UINT:
            depth_aspect_mask |= vk.VK_IMAGE_ASPECT_STENCIL_BIT
        
        attachments["depth"] = self._createImageView(depth_format, depth_usage, depth_aspect_mask)
        
        return attachments

    def _createRenderPass(self, color_format, depth_format):
        # Vulkan: create color and depth attachments
        color_attachement_desc = vk.VkAttachmentDescription(
            format=color_format,
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
            format=depth_format,
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
            colorAttachmentCount=1,
            pColorAttachments=[color_reference],
            pDepthStencilAttachment=depth_reference,
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
        
        return vk.vkCreateRenderPass(device=self._logical_device,
                                     pCreateInfo=render_pass_info,
                                     pAllocator=None)

    def _createFramebuffer(self, render_pass, color_image_view, depth_image_view):
        framebuffer_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=render_pass,
            attachmentCount=2,
            pAttachments=[color_image_view, depth_image_view],
            width=self._width,
			height=self._height,
			layers=1,
            flags=0
        )
        
        return vk.vkCreateFramebuffer(device=self._logical_device,
                                      pCreateInfo=framebuffer_info,
                                      pAllocator=None)

    def _createGraphicsPipeline(self, render_pass):
        # Load in vertex and fragment shaders
        vert_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.vert.spv")
        frag_shader_module = self._createShaderModule("./shaders/compiled/vertex_color.frag.spv")
        print("Vk Create Shader Modules: success")
        
        # Vulkan: set up shader stage information
        vertex_shader_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            module=vert_shader_module,
            pName="main",
            pSpecializationInfo=None,
            flags=0
        )

        fragment_shader_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=frag_shader_module,
            pName="main",
            pSpecializationInfo=None,
            flags=0
        )
        
        # Vulkan: MVP matrix via push constant block
        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            size=glm.sizeof(glm.mat4),
            offset=0
        )
        
        # Vulkan: pipeline layout information
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=0,
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range],
            flags=0
        )
        
        # Vulkan: create pipeline layout
        pipeline_layout = vk.vkCreatePipelineLayout(device=self._logical_device,
                                                    pCreateInfo=pipeline_layout_info,
                                                    pAllocator=None)

        # Vulkan: create pipeline cache information
        pipeline_cache_info = vk.VkPipelineCacheCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
        )
        
        # Vulkan: create pipeline cache
        pipeline_cache = vk.vkCreatePipelineCache(device=self._logical_device,
                                                  pCreateInfo=pipeline_cache_info,
                                                  pAllocator=None)

        # Vulkan: input assemble information
        input_assembly_info = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
            flags=0
        )

        # Vulkan: rasterization information
        rasterization_info = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            cullMode=vk.VK_CULL_MODE_NONE, #vk.VK_CULL_MODE_BACK_BIT,
		    frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            lineWidth=1.0,
            flags=0
        )

        # Vulkan: color blending information
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=vk.VK_FALSE,
            colorWriteMask=0
        )
        color_blend_info = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_blend_attachment],
            flags=0
        )

        # Vulkan: depth/stencil information
        depth_stencil_info = vk.VkPipelineDepthStencilStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_TRUE,
            depthCompareOp=vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            back=vk.VkStencilOpState(compareOp=vk.VK_COMPARE_OP_ALWAYS),
            flags=0
        )

        # Vulkan: viewport and scissor information
        dynamic_state_info = vk.VkPipelineDynamicStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=2,
            pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR],
            flags=0
        )
        viewport_info = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            scissorCount=1,
            flags=0
        )

        # Vulkan: multisampling information
        multisample_info = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            flags=0
        )

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

        # Vulkan: vertex input state information
        vertex_input_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[vertex_binding_desc],
            vertexAttributeDescriptionCount=2,
            pVertexAttributeDescriptions=[vertex_position_attrib, vertex_color_attrib],
            flags=0
        )

        # Vulkan: graphics pipeline information
        graphics_pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            layout=pipeline_layout,
            renderPass=render_pass,
            basePipelineIndex=-1,
            basePipelineHandle=vk.VK_NULL_HANDLE,
            pInputAssemblyState=input_assembly_info,
            pRasterizationState=rasterization_info,
            pColorBlendState=color_blend_info,
            pMultisampleState=multisample_info,
            pViewportState=viewport_info,
            pDepthStencilState=depth_stencil_info,
            pDynamicState=dynamic_state_info,
            stageCount=2,
            pStages=[vertex_shader_info, fragment_shader_info],
            pVertexInputState=vertex_input_info,
            flags=0
        )

        # Vulkan: create graphics pipeline
        graphics_pipeline = vk.vkCreateGraphicsPipelines(device=self._logical_device,
                                                         pipelineCache=pipeline_cache,
                                                         createInfoCount=1,
                                                         pCreateInfos=[graphics_pipeline_info],
                                                         pAllocator=None)[0]

        # Vulkan: free shader modules
        vk.vkDestroyShaderModule(device=self._logical_device, shaderModule=vert_shader_module, pAllocator=None)
        vk.vkDestroyShaderModule(device=self._logical_device, shaderModule=frag_shader_module, pAllocator=None)
        
        return {"pipeline": graphics_pipeline, "layout": pipeline_layout}

    def _createGraphicCommandBuffer(self, render_pass, framebuffer, graphics_pipeline, vertex_object):
        # Vulkan: command buffer allocation information
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        # Vulkan: allocate command buffer
        self._render_cmd_buffer = vk.vkAllocateCommandBuffers(device=self._logical_device,
                                                              pAllocateInfo=command_buffer_info)[0]

        # Vulkan: prepare data for recording command buffers
        command_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        
        # Vulkan: record command buffer
        vk.vkBeginCommandBuffer(commandBuffer=self._render_cmd_buffer, pBeginInfo=command_begin_info)
        
        # Vulkan: set clear color and depth values
        clear_color = vk.VkClearValue(
			color=vk.VkClearColorValue(float32=[0.4, 0.1, 0.6, 1.0]) # R,G,B,A
        )
        clear_depth = vk.VkClearValue(
            depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)
		)
        
        # Vulkan: render pass begin information
        render_pass_begin_info = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderArea=vk.VkRect2D(extent=vk.VkExtent2D(width=self._width, height=self._height)),
            clearValueCount=2,
            pClearValues=[clear_color, clear_depth],
            renderPass=render_pass,
            framebuffer=framebuffer
        )
        
        # Vulkan: begin render pass
        vk.vkCmdBeginRenderPass(commandBuffer=self._render_cmd_buffer,
                                pRenderPassBegin=render_pass_begin_info,
                                contents=vk.VK_SUBPASS_CONTENTS_INLINE)

        # Vulkan: update dynamic viewport and scissor state
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=self._width,
            height=self._height,
            minDepth=0.0,
            maxDepth=1.0
        )
        vk.vkCmdSetViewport(commandBuffer=self._render_cmd_buffer,
                            firstViewport=0,
                            viewportCount=1,
                            pViewports=[viewport])
        
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=vk.VkExtent2D(width=self._width, height=self._height)
        )
        vk.vkCmdSetScissor(commandBuffer=self._render_cmd_buffer,
                           firstScissor=0,
                           scissorCount=1,
                           pScissors=[scissor])

        # Vulkan: bind pipeline
        vk.vkCmdBindPipeline(commandBuffer=self._render_cmd_buffer,
                             pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline=graphics_pipeline["pipeline"])
        
        # Vulkan: render scene
        mvp = ffi.cast("float *", ffi.from_buffer(self._mvp_transform))
        #mvp_ffi_buffer = ffi.buffer(mvp, glm.sizeof(glm.mat4))
        #print(mvp_ffi_buffer)
        #mvp_array = np.frombuffer(mvp_ffi_buffer, dtype=np.float32, count=16, offset=0)
        #print(mvp_array)
        
        print(vertex_object["vertices"]["buffer"], vertex_object["indices"]["buffer"])
        vk.vkCmdBindVertexBuffers(commandBuffer=self._render_cmd_buffer,
                                  firstBinding=0,
                                  bindingCount=1,
                                  pBuffers=[vertex_object["vertices"]["buffer"]],
                                  pOffsets=[0])
        vk.vkCmdBindIndexBuffer(commandBuffer=self._render_cmd_buffer,
                                buffer=vertex_object["indices"]["buffer"],
                                offset=0,
                                indexType=vk.VK_INDEX_TYPE_UINT32)
        vk.vkCmdPushConstants(commandBuffer=self._render_cmd_buffer,
                              layout=graphics_pipeline["layout"],
                              stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                              offset=0,
                              size=glm.sizeof(glm.mat4),
                              pValues=mvp)

        #vk.vkCmdDrawIndexed(commandBuffer=self._render_cmd_buffer,
        #                    indexCount=3,
        #                    instanceCount=1,
        #                    firstIndex=0,
        #                    vertexOffset=0,
        #                    firstInstance=0)
        vk.vkCmdDraw(commandBuffer=self._render_cmd_buffer,
                     vertexCount=3,
                     instanceCount=1,
                     firstVertex=0,
                     firstInstance=0)
        
        # Vulkan: end render pass and command buffer
        vk.vkCmdEndRenderPass(commandBuffer=self._render_cmd_buffer)
        vk.vkEndCommandBuffer(commandBuffer=self._render_cmd_buffer)
        
        # Vulkan: destroy pipeline layout (no longer needed)
        vk.vkDestroyPipelineLayout(device=self._logical_device,
                                   pipelineLayout=graphics_pipeline["layout"],
                                   pAllocator=None)
        graphics_pipeline["layout"] = None

    def _findMemoryTypeIndex(self, supported_memory_indices, req_properties):
        for i in range(self._device_mem_props.memoryTypeCount):
            supported = supported_memory_indices & (1 << i)
            sufficient = (self._device_mem_props.memoryTypes[i].propertyFlags & req_properties) == req_properties
            if supported and sufficient:
                return i
        return -1

    def _createBuffer(self, usage_flags, memory_prop_flags, size, data=None):
        # Vulkan: create buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage_flags,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        buffer = vk.vkCreateBuffer(device=self._logical_device,
                                   pCreateInfo=buffer_info,
                                   pAllocator=None)
        
        # Vulkan: get buffer memory requirments
        mem_reqs = vk.vkGetBufferMemoryRequirements(device=self._logical_device,
                                                    buffer=buffer)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits, memory_prop_flags)
        
        # Vulkan: allocate memory
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        memory = vk.vkAllocateMemory(device=self._logical_device,
                                     pAllocateInfo=mem_alloc_info,
                                     pAllocator=None)
        
        # Vulkan: copy data (if not None)
        if data is not None:
            memory_location = vk.vkMapMemory(device=self._logical_device,
                                             memory=memory,
                                             offset=0,
                                             size=size,
                                             flags=0)
            
            ffi.memmove(memory_location, data, size)
            #TEST
            #memory_array = np.frombuffer(memory_location, dtype=data.dtype, count=data.size, offset=0)
            #print(memory_array)
            
            vk.vkUnmapMemory(device=self._logical_device, memory=memory)

        
        # Vulkan: bind buffer memory
        vk.vkBindBufferMemory(device=self._logical_device,
                              buffer=buffer,
                              memory=memory,
                              memoryOffset=0)
        
        return {"buffer": buffer, "memory": memory}

    def _getSupportedDepthFormat(self, physical_device):
        # Vulkan: find highest precision packed format that is supported by device
        possible_formats = [
            vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
            vk.VK_FORMAT_D32_SFLOAT,
            vk.VK_FORMAT_D24_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM
        ]
        for fmt in possible_formats:
            format_props = vk.vkGetPhysicalDeviceFormatProperties(physicalDevice=physical_device,
                                                                  format=fmt)
            if format_props.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return fmt
        
        print("Vk: WARNING> No suitable depth format supported")
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
            ),
            flags=0
        )
        image_view = vk.vkCreateImageView(device=self._logical_device,
                                          pCreateInfo=image_view_info,
                                          pAllocator=None)

        return {"image": image, "memory": memory, "view": image_view}

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

    def _submitWork(self, command_buffer, queue):
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )

        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=0
        )
        fence = vk.vkCreateFence(device=self._logical_device,
                                 pCreateInfo=fence_info,
                                 pAllocator=None)

        vk.vkQueueSubmit(queue=queue,
                         submitCount=1, 
                         pSubmits=[submit_info],
                         fence=fence)
        vk.vkWaitForFences(device=self._logical_device,
                           fenceCount=1,
                           pFences=[fence],
                           waitAll=vk.VK_TRUE,
                           timeout=np.iinfo(np.uint64).max)
        vk.vkDestroyFence(device=self._logical_device,
                          fence=fence,
                          pAllocator=None)
        if command_buffer == self._render_cmd_buffer:
            print("render submitted")

    def _instertImageMemoryBarrier(self, command_buffer, image, src_access_mask, dst_access_mask,
                                   old_img_layout, new_img_layout, src_stage_mask, dst_stage_mask,
                                   subresource_range):
        memory_barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            srcAccessMask=src_access_mask,
            dstAccessMask=dst_access_mask,
            oldLayout=old_img_layout,
            newLayout=new_img_layout,
            image=image,
            subresourceRange=subresource_range
        )
        vk.vkCmdPipelineBarrier(commandBuffer=command_buffer,
				                srcStageMask=src_stage_mask,
				                dstStageMask=dst_stage_mask,
				                dependencyFlags=0,
				                memoryBarrierCount=0,
                                pMemoryBarriers=[],
				                bufferMemoryBarrierCount=0,
                                pBufferMemoryBarriers=[],
                                imageMemoryBarrierCount=1,
                                pImageMemoryBarriers=[memory_barrier])

    def _getRawImage(self):
        # Vulkan: create destination image to copy to and read the memory from
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
			format=vk.VK_FORMAT_R8G8B8A8_UNORM,
			extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1),
			arrayLayers=1,
			mipLevels=1,
			initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
			samples=vk.VK_SAMPLE_COUNT_1_BIT,
			tiling=vk.VK_IMAGE_TILING_LINEAR,
			usage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            flags=0
        )
        image = vk.vkCreateImage(device=self._logical_device,
                                 pCreateInfo=image_info,
                                 pAllocator=None)
        
        # Vulkan: create and bind memory for the image
        mem_reqs = vk.vkGetImageMemoryRequirements(device=self._logical_device, image=image)
        mem_type_index = self._findMemoryTypeIndex(mem_reqs.memoryTypeBits,
                                                   vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        mem_alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index
        )
        image_memory = vk.vkAllocateMemory(device=self._logical_device,
                                           pAllocateInfo=mem_alloc_info,
                                           pAllocator=None)
        vk.vkBindImageMemory(device=self._logical_device, image=image, memory=image_memory, memoryOffset=0)

        # Vulkan: copy rendered image to host visible destination image
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        copy_command_buffer = vk.vkAllocateCommandBuffers(self._logical_device, command_buffer_info)[0]
        
        buffer_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=0
        )
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=buffer_begin_info)

        subresource_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        self._instertImageMemoryBarrier(copy_command_buffer, image, 0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                        vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        subresource_range)
        image_copy_region = vk.VkImageCopy(
            srcSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            dstSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1)
        )
        vk.vkCmdCopyImage(commandBuffer=copy_command_buffer,
                          srcImage=self._color_attachment["image"],
				          srcImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dstImage=image,
                          dstImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          regionCount=1,
                          pRegions=[image_copy_region])
        self._instertImageMemoryBarrier(copy_command_buffer, image, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                        vk.VK_ACCESS_MEMORY_READ_BIT, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, subresource_range)

        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)
        
        self._submitWork(copy_command_buffer, self._graphic_queue)
        
        # Vulkan: get layout of the image (including row pitch)
        subresource = vk.VkImageSubresource(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT)
        subresource_layout = vk.vkGetImageSubresourceLayout(device=self._logical_device,
                                                            image=image,
                                                            pSubresource=[subresource])
        
        # Vulkan: map image memory so we can read it
        memory_location = vk.vkMapMemory(device=self._logical_device,
                                         memory=image_memory,
                                         offset=0,
                                         size=self._width * self._height * 4,
                                         flags=0)
        
        # Convert memory to numpy array and extract RGB data
        memory_array = np.frombuffer(memory_location, dtype=np.uint8, count=self._width * self._height * 4,
                                     offset=subresource_layout.offset)
        print(f"FIRST PX: {memory_array[0]} {memory_array[1]} {memory_array[2]} {memory_array[3]}")
        for px in range(0, memory_array.size, 4):
            if memory_array[px+0] != 102 or memory_array[px+1] != 25 or memory_array[px+2] != 153:
                print(f"FOUND non-purple px: {px} ({memory_array[px+0]}, {memory_array[px+1]}, {memory_array[px+3]})")
                break
        rgb_img = memory_array[np.mod(np.arange(memory_array.size), 4) != 3]
        
        # TEST -> save PPM image
        file = open("vk_image.ppm", "wb")
        file.write(f"P6\n{self._width} {self._height}\n255\n".encode("utf-8"))
        file.write(rgb_img.tobytes())
        file.close()

        return rgb_img

    def _getJpegImage(self):
        pass

    def _getH264VideoFrame(self):
        pass

    def _messageSeverityToString(self, message_severity):
        severity_str = "UNKNOWN"
        if message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            severity_str = "VERBOSE"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            severity_str = "INFO"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity_str = "WARNING"
        elif message_severity == vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity_str = "ERROR"
        return severity_str
    
    def _debugMessageCallback(self, message_severity, message_types, callback_data, user_data):
        message = ffi.string(callback_data.pMessage).decode('utf-8')
        severity = self._messageSeverityToString(message_severity)
        print(f"[VULKAN {severity}]: {message}")
        return vk.VK_FALSE
"""