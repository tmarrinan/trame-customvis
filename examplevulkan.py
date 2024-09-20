import time
import math
import asyncio
import glm
import cv2
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
        self._velocity_x = 50
        self._velocity_y = 30
        # User interaction with triangle
        self._triangle_selected = False
        # Trame image options
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}
        self._gpu_video_encode = False
        # Animation timing
        self._ready = False
        self._start_time = round(time.time_ns() / 1000000)
        self._prev_time = self._start_time
        # Vulkan variables
        self._vk = {}
        
        # Initialize
        self._initVulkan()

    def getSize(self):
        return (self._width, self._height)

    def setSize(self, width, height):
        # Not ready to render until framebuffer recreated
        self._ready = False
        
        # Allow device commands to finish
        vk.vkDeviceWaitIdle(device=self._vk["device"])

        # Destroy old framebuffer and image views
        vk.vkDestroyFramebuffer(device=self._vk["device"], framebuffer=self._vk["framebuffer"], pAllocator=None)

        vk.vkDestroyImageView(device=self._vk["device"], imageView=self._vk["color_attachment_view"], pAllocator=None)
        vk.vkDestroyImage(device=self._vk["device"], image=self._vk["color_attachment_image"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=self._vk["color_attachment_memory"], pAllocator=None)

        vk.vkDestroyImageView(device=self._vk["device"], imageView=self._vk["depth_attachment_view"], pAllocator=None)
        vk.vkDestroyImage(device=self._vk["device"], image=self._vk["depth_attachment_image"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=self._vk["depth_attachment_memory"], pAllocator=None)
        
        vk.vkDestroyImageView(device=self._vk["device"], imageView=self._vk["compute_rgb_view"], pAllocator=None)
        vk.vkDestroyImage(device=self._vk["device"], image=self._vk["compute_rgb_image"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=self._vk["compute_rgb_memory"], pAllocator=None)

        # Set new size
        self._width = width
        self._height = height

        # Create new framebuffer and image views
        self._vk.update(self._createColorAttachment(self._vk["color_format"]))
        self._vk.update(self._createDepthAttachment(self._vk["depth_format"]))
        self._vk["framebuffer"] = self._createFramebuffer()
        
        # Create new compute image views
        compute_image = self._createImageView(self._width * 3, self._height, vk.VK_FORMAT_R8_UNORM,
                                              vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                              vk.VK_IMAGE_ASPECT_COLOR_BIT)
        self._vk["compute_rgb_image"] = compute_image["image"]
        self._vk["compute_rgb_memory"] = compute_image["memory"]
        self._vk["compute_rgb_view"] = compute_image["view"]
        
        # Reconfigure compute descriptor set with new framebuffer color attachment
        self._configureComputeDescriptorSet()
        
        # Re-record the draw and compute commands
        vk.vkResetCommandBuffer(commandBuffer=self._vk["graphic_cmd_buffer"], flags=0)
        vk.vkResetCommandBuffer(commandBuffer=self._vk["compute_cmd_buffer"], flags=0)
        self._recordDrawCommands()
        self._recordComputeRgbaToRgbCommands()
        
        # Now ready to render again
        self._ready = True

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
        if not self._ready:
            return None
        elif self._image_type == "rgb":
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
        if not self._ready:
            return

        # Animate
        now = time.time_ns() / 1000000
        """
        dt = (now - self._prev_time) / 1000
        
        dx = self._velocity_x * dt
        dy = self._velocity_y * dt
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
            self._velocity_y *= -1
        else:
            self._triangle_center[1] += dy
        """
        
        # Update model, view, and projection matrices
        pos = glm.vec3(self._triangle_center[0], self._triangle_center[1], -1.5)
        proj = glm.ortho(0, self._width, 0, self._height, 1, 2)
        view = glm.mat4(1.0)
        model = glm.translate(glm.mat4(1.0), pos) * glm.scale(glm.mat4(1.0), glm.vec3(50.0, 50.0, 1.0))
        
        mat_model = np.ctypeslib.as_array(glm.value_ptr(model), shape=(16,))
        mat_view = np.ctypeslib.as_array(glm.value_ptr(view), shape=(16,))
        mat_proj = np.ctypeslib.as_array(glm.value_ptr(proj), shape=(16,))
        
        #ubo_data = np.array([mat_model, mat_view, mat_proj], dtype=np.float32)
        #ffi.memmove(self._vk["uniform_memory_location"], ubo_data, ubo_data.nbytes)
        
        # Make class member var (only allocate once)
        ubo_data = np.empty(48, dtype=np.float32)
        ubo_data[0:16] = mat_model
        ubo_data[16:32] = mat_view
        ubo_data[32:48] = mat_proj
        ffi.memmove(self._vk["graphic_uniform_memory_loc"], ubo_data, ubo_data.nbytes)
        
        # Vulkan: submit render command to graphics queue
        self._submitWork(self._vk["graphic_cmd_buffer"], self._vk["graphic_compute_queue"])
        #vk.vkDeviceWaitIdle(device=self._vk["device"])

        # Update render time
        self._prev_time = now

    """
    Handler for left mouse button
    return: whether or not rerender is required
    """
    def onLeftMouseButton(self, mouse_x, mouse_y, pressed):
        if pressed:
            if self._distanceSqr2d((mouse_x, self._height - mouse_y), self._triangle_center) < (50.0 ** 2):
                self._triangle_selected = True
        else:
            self._triangle_selected = False
        return False

    """
    Handler for right mouse button
    return: whether or not rerender is required
    """
    def onRightMouseButton(self, mouse_x, mouse_y, pressed):
        return False

    """
    Handler for mouse movement
    return: whether or not rerender is required
    """
    def onMouseMove(self, mouse_x, mouse_y):
        rerender = False
        if self._triangle_selected:
            self._triangle_center[0] = mouse_x
            self._triangle_center[1] = self._height - mouse_y
            rerender = True
        return rerender

    """
    Handler for mouse scroll wheel
    return: whether or not rerender is required
    """
    def onMouseWheel(self, mouse_x, mouse_y, delta):
        return False

    """
    Handler for keyboard key pressed down
    return: whether or not rerender is required
    """
    def onKeyDown(self, key):
        return False

    """
    Handler for keyboard key released up
    return: whether or not rerender is required
    """
    def onKeyUp(self, key):
        return False

    def _initVulkan(self):
        # Create Vulkan instance and find a physical rendering device
        self._vk["instance"] = self._createInstance()
        self._vk["physical_device"] = self._findPhysicalDevice()

        # Select framebuffer color and depth formats
        self._vk["color_format"] = vk.VK_FORMAT_R8G8B8A8_UNORM
        self._vk["depth_format"] = self._getSupportedDepthFormat()

        # Create a logical device and graphics queue
        self._vk["graphic_compute_fam_idx"] = self._getGraphicComputeQueueFamilyIndex()
        self._vk["device"] = self._createLogicalDevice()
        self._vk["graphic_compute_queue"] = self._getGraphicComputeQueue()
        
        # Create the uniform buffer object for passing data to our shaders
        self._vk.update(self._createGraphicsUniformBuffer())
        
        # Set up the render pass and graphics pipeline
        self._vk["render_pass"] = self._createRenderPass(self._vk["color_format"], self._vk["depth_format"])
        self._vk.update(self._createGraphicsPipeline())
        
        # Create framebuffer
        self._vk.update(self._createColorAttachment(self._vk["color_format"]))
        self._vk.update(self._createDepthAttachment(self._vk["depth_format"]))
        self._vk["framebuffer"] = self._createFramebuffer()
        
        # Create the graphic command buffer
        self._vk["command_pool"] = self._createCommandPool()
        self._vk["graphic_cmd_buffer"] = self._createCommandBuffer()

        # Create the vertex buffer object for our triangle mesh
        self._vk.update(self._createTriangleMesh())
        
        # Set up compute shader pipeline for converting rendered image to RGB
        self._vk["compute_cmd_buffer"] = self._createCommandBuffer()
        self._vk.update(self._createComputeUniformBuffer())
        self._configureComputeDescriptorSet()
        self._vk.update(self._createComputePipeline())

        # Print out all Vulkan variables
        print("Vulkan: application objects")
        for key,item in self._vk.items():
            print(f"  {key:26s}: {item}")

        # Record the draw commands
        self._recordDrawCommands()
        
        # Record the compute commands
        self._recordComputeRgbaToRgbCommands()
        
        # Now ready to render
        self._ready = True

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

    def _getGraphicComputeQueueFamilyIndex(self):
        # Find index for a graphics queue
        queue_family_index = -1
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self._vk["physical_device"])
        for i in range(len(queue_families)):
            is_graphic = queue_families[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
            is_compute = queue_families[i].queueFlags & vk.VK_QUEUE_COMPUTE_BIT
            if queue_families[i].queueCount > 0 and is_graphic and is_compute:
                queue_family_index = i
        if queue_family_index < 0:
            print(f"Vulkan: WARNING> no graphic/compute queue family found")

        print(f"queue family index: {queue_family_index}")
        return queue_family_index

    def _createLogicalDevice(self):
        # Create queue information
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self._vk["graphic_compute_fam_idx"],
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

    def _getGraphicComputeQueue(self):
        # Get graphics queue
        return vk.vkGetDeviceQueue(device=self._vk["device"],
                                   queueFamilyIndex=self._vk["graphic_compute_fam_idx"],
                                   queueIndex=0)

    def _createGraphicsUniformBuffer(self):
        # Create descriptor set layout information
        layout_binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT
        )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=[layout_binding]
        )
        
        # Create descriptor set layout
        layout = vk.vkCreateDescriptorSetLayout(device=self._vk["device"], pCreateInfo=layout_info, pAllocator=None)

        # Create descriptor pool information
        pool_size = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1)
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
            maxSets=1
        )
        
        # Create descriptor pool
        pool = vk.vkCreateDescriptorPool(device=self._vk["device"], pCreateInfo=pool_info, pAllocator=None)

        # Allocate descriptor set
        allocate_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=pool,
            descriptorSetCount=1, 
            pSetLayouts=[layout]
        )
        
        # Create descriptor set
        descriptor_set = vk.vkAllocateDescriptorSets(device=self._vk["device"], pAllocateInfo=allocate_info)[0]
        

        # Create buffer
        buffer_size = 3 * glm.sizeof(glm.mat4)
        uniform_buffer = self._createBuffer(buffer_size, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Map memory location
        memory_location = vk.vkMapMemory(device=self._vk["device"], 
                                         memory=uniform_buffer["memory"], 
                                         offset=0,
                                         size=buffer_size,
                                         flags=0)

        # Configure descriptor set
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=uniform_buffer["buffer"],
            offset=0,
            range=buffer_size
        )
        descriptor_write = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=descriptor_set,
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=[buffer_info]
        )
        vk.vkUpdateDescriptorSets(device=self._vk["device"], descriptorWriteCount=1,
                                 pDescriptorWrites=[descriptor_write], descriptorCopyCount=0,
                                 pDescriptorCopies=None)

        return {"graphic_set": descriptor_set,
                "graphic_set_layout": layout,
                "graphic_uniform_buffer": uniform_buffer["buffer"],
                "graphic_uniform_memory": uniform_buffer["memory"],
                "graphic_uniform_memory_loc": memory_location}

    def _createComputeUniformBuffer(self):
        # Create descriptor set layout information
        layout_binding_ubo = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
        )
        layout_binding_img_in = vk.VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
        )
        layout_binding_img_out = vk.VkDescriptorSetLayoutBinding(
            binding=2,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
        )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=3,
            pBindings=[layout_binding_ubo, layout_binding_img_in, layout_binding_img_out]
        )
        
        # Create descriptor set layout
        layout = vk.vkCreateDescriptorSetLayout(device=self._vk["device"], pCreateInfo=layout_info, pAllocator=None)

        # Create descriptor pool information
        pool_size_ubo = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1)
        pool_size_img_in = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, descriptorCount=1)
        pool_size_img_out = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=1)
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=3,
            pPoolSizes=[pool_size_ubo, pool_size_img_in, pool_size_img_out],
            maxSets=1
        )
        
        # Create descriptor pool
        pool = vk.vkCreateDescriptorPool(device=self._vk["device"], pCreateInfo=pool_info, pAllocator=None)

        # Allocate descriptor set
        allocate_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=pool,
            descriptorSetCount=1, 
            pSetLayouts=[layout]
        )
        
        # Create descriptor set
        descriptor_set = vk.vkAllocateDescriptorSets(device=self._vk["device"], pAllocateInfo=allocate_info)[0]
        
        # Create ubo buffer
        buffer_size = glm.sizeof(glm.uvec2)
        uniform_buffer = self._createBuffer(buffer_size, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Map memory location
        memory_location = vk.vkMapMemory(device=self._vk["device"], 
                                         memory=uniform_buffer["memory"], 
                                         offset=0,
                                         size=buffer_size,
                                         flags=0)
        
        # Create out images
        uniform_image_out = self._createImageView(self._width * 3, self._height, vk.VK_FORMAT_R8_UNORM,
                                                  vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                  vk.VK_IMAGE_ASPECT_COLOR_BIT)

        return {"compute_set": descriptor_set,
                "compute_set_layout": layout,
                "compute_uniform_buffer": uniform_buffer["buffer"],
                "compute_uniform_memory": uniform_buffer["memory"],
                "compute_uniform_memory_loc": memory_location,
                "compute_rgb_image": uniform_image_out["image"],
                "compute_rgb_memory": uniform_image_out["memory"],
                "compute_rgb_view": uniform_image_out["view"]}

    def _configureComputeDescriptorSet(self):
        # Configure descriptor set
        buffer_size = glm.sizeof(glm.uvec2)
        buffer_info_ubo = vk.VkDescriptorBufferInfo(
            buffer=self._vk["compute_uniform_buffer"],
            offset=0,
            range=buffer_size
        )
        descriptor_write_ubo = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self._vk["compute_set"],
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=[buffer_info_ubo]
        )
        image_info_in = vk.VkDescriptorImageInfo(
            sampler=None,
            imageView=self._vk["color_attachment_view"],
            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        )
        descriptor_write_img_in = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self._vk["compute_set"],
            dstBinding=1,
            dstArrayElement=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            pImageInfo=[image_info_in]
        )
        image_info_out = vk.VkDescriptorImageInfo(
            sampler=None,
            imageView=self._vk["compute_rgb_view"],
            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        )
        descriptor_write_img_out = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self._vk["compute_set"],
            dstBinding=2,
            dstArrayElement=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            pImageInfo=[image_info_out]
        )
        vk.vkUpdateDescriptorSets(device=self._vk["device"],
                                  descriptorWriteCount=3,
                                  pDescriptorWrites=[descriptor_write_ubo, descriptor_write_img_in, descriptor_write_img_out],
                                  descriptorCopyCount=0,
                                  pDescriptorCopies=None)

    def _createRenderPass(self, color_format, depth_format):
        # Create color attachment description
        color_attachment_desc = vk.VkAttachmentDescription(
            format=color_format,
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
        
        # Create depth attachment description
        depth_attachement_desc = vk.VkAttachmentDescription(
            format=depth_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        
        # Create depth attachment reference
        depth_attachement_ref = vk.VkAttachmentReference(
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL 
        )

        # Create subpass description
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=color_attachment_ref,
            pDepthStencilAttachment=depth_attachement_ref
        )

        # Create render pass information
        render_pass_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=2,
            pAttachments=[color_attachment_desc, depth_attachement_desc],
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
        
        # Dynamic viewport and scissor information
        dynamic_state_info = vk.VkPipelineDynamicStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=2,
            pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR],
            flags=0
        )
        viewport_state_info = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            scissorCount=1,
            flags=0
        )

        # Create rasterization state information
        raterization_state_info = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
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

        # Create depth/stencil information
        depth_stencil_info = vk.VkPipelineDepthStencilStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_TRUE,
            depthCompareOp=vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            back=vk.VkStencilOpState(compareOp=vk.VK_COMPARE_OP_ALWAYS)
        )
        
        # Set up push constant
        push_constant_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                                                     offset=0,
                                                     size=64)

        # Create pipeline layout information
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self._vk["graphic_set_layout"]]
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
            pDepthStencilState=depth_stencil_info,
            pColorBlendState=color_blend_info,
            pDynamicState=dynamic_state_info,
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

        return {"graphic_layout": pipeline_layout, "graphic_pipeline": graphics_pipeline}

    def _createComputePipeline(self):
        # Load in compute shader
        comp_shader_module = self._createShaderModule("./shaders/compiled/rgba2rgb.comp.spv")
        
        # Set up shader stage information
        compute_shader_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=comp_shader_module,
            pName="main"
        )
        
        # Create pipeline layout information
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self._vk["compute_set_layout"]]
        )

        # Create pipeline layout
        pipeline_layout = vk.vkCreatePipelineLayout(device=self._vk["device"],
                                                    pCreateInfo=pipeline_layout_info,
                                                    pAllocator=None)
        
        # Create compute pipeline information
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=compute_shader_info,
            layout=pipeline_layout,
            flags=0
        )
        
        # Create compute pipeline
        compute_pipeline = vk.vkCreateComputePipelines(device=self._vk["device"],
                                                       pipelineCache=vk.VK_NULL_HANDLE,
                                                       createInfoCount=1,
                                                       pCreateInfos=pipeline_info,
                                                       pAllocator=None)[0]

        # Vulkan: free shader modules
        vk.vkDestroyShaderModule(device=self._vk["device"], shaderModule=comp_shader_module, pAllocator=None)

        return {"compute_layout": pipeline_layout, "compute_pipeline": compute_pipeline}

    def _createColorAttachment(self, color_format):
        # Create color image view for framebuffer
        color_attachment = self._createImageView(self._width, self._height, color_format,
                                                 vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                 vk.VK_IMAGE_ASPECT_COLOR_BIT)

        return {"color_attachment_image": color_attachment["image"], "color_attachment_memory": color_attachment["memory"],
                "color_attachment_view": color_attachment["view"]}

    def _createDepthAttachment(self, depth_format):
        # Create depth image view for framebuffer
        depth_aspect_mask = vk.VK_IMAGE_ASPECT_DEPTH_BIT
        if depth_format >= vk.VK_FORMAT_D16_UNORM_S8_UINT:
            depth_aspect_mask |= vk.VK_IMAGE_ASPECT_STENCIL_BIT

        depth_attachment = self._createImageView(self._width, self._height, depth_format,
                                                 vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depth_aspect_mask)

        return {"depth_attachment_image": depth_attachment["image"], "depth_attachment_memory": depth_attachment["memory"],
                "depth_attachment_view": depth_attachment["view"]}

    def _createFramebuffer(self):
        # Create framebuffer information
        framebuffer_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self._vk["render_pass"],
            attachmentCount=2,
            pAttachments=[self._vk["color_attachment_view"], self._vk["depth_attachment_view"]],
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
            queueFamilyIndex=self._vk["graphic_compute_fam_idx"],
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
        vertices = np.array([ 0.0, -1.0, 1.0, 0.0, 0.0,
                             -1.0,  1.0, 0.0, 1.0, 0.0,
                              1.0,  1.0, 0.0, 0.0, 1.0],
                            dtype=np.float32)

        # Create command buffer for copy operation
        command_buffer_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._vk["command_pool"],
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        copy_command_buffer = vk.vkAllocateCommandBuffers(self._vk["device"], command_buffer_info)[0]
        
        # Create buffer for triangle vertex data
        staging = self._createBuffer(vertices.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        vertex_buffer = self._createBuffer(vertices.nbytes,
                                           vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        # Transfer vertex data to staging buffer
        memory_location = vk.vkMapMemory(device=self._vk["device"], memory=staging["memory"],
                                         offset=0, size=vertices.nbytes, flags=0)
        ffi.memmove(memory_location, vertices, vertices.nbytes)
        vk.vkUnmapMemory(device=self._vk["device"], memory=staging["memory"])
        
        # Transfer vertex data to GPU buffer
        copy_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        vk.vkBeginCommandBuffer(commandBuffer=copy_command_buffer, pBeginInfo=copy_begin_info)
        copy_region = vk.VkBufferCopy(size=vertices.nbytes)
        vk.vkCmdCopyBuffer(commandBuffer=copy_command_buffer,
                           srcBuffer=staging["buffer"],
                           dstBuffer=vertex_buffer["buffer"],
                           regionCount=1,
                           pRegions=[copy_region])
        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)
        
        self._submitWork(copy_command_buffer, self._vk["graphic_compute_queue"])
        
        # Free stagin buffer
        vk.vkDestroyBuffer(device=self._vk["device"], buffer=staging["buffer"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=staging["memory"], pAllocator=None)
        
        # Free copy command buffer
        vk.vkFreeCommandBuffers(device=self._vk["device"],
                                commandPool=self._vk["command_pool"],
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])

        return {"vbo_vert_buffer": vertex_buffer["buffer"], "vbo_vert_memory": vertex_buffer["memory"]}

    def _recordDrawCommands(self):
        # Prepare data for recording command buffers
        command_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        
        # Record command buffer
        vk.vkBeginCommandBuffer(commandBuffer=self._vk["graphic_cmd_buffer"], pBeginInfo=command_begin_info)
        
        # Set clear color and depth values
        clear_color = vk.VkClearValue(
			color=vk.VkClearColorValue(float32=[0.4, 0.1, 0.6, 1.0]) # R,G,B,A
        )
        clear_depth = vk.VkClearValue(
			depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)
        )
        
        # Render pass begin information
        render_pass_begin_info = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderArea=[[0, 0], [self._width, self._height]], #vk.VkRect2D(extent=vk.VkExtent2D(width=self._width, height=self._height)),
            clearValueCount=2,
            pClearValues=[clear_color, clear_depth],
            renderPass=self._vk["render_pass"],
            framebuffer=self._vk["framebuffer"]
        )
        
        # Begin render pass
        vk.vkCmdBeginRenderPass(commandBuffer=self._vk["graphic_cmd_buffer"],
                                pRenderPassBegin=render_pass_begin_info,
                                contents=vk.VK_SUBPASS_CONTENTS_INLINE)

        # Update dynamic viewport and scissor state
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=self._width,
            height=self._height,
            minDepth=0.0,
            maxDepth=1.0
        )
        vk.vkCmdSetViewport(commandBuffer=self._vk["graphic_cmd_buffer"], firstViewport=0,
                            viewportCount=1, pViewports=[viewport])
        
        scissor = vk.VkRect2D(
            offset=[0, 0],
            extent=[self._width, self._height]
        )
        vk.vkCmdSetScissor(commandBuffer=self._vk["graphic_cmd_buffer"], firstScissor=0,
                           scissorCount=1, pScissors=[scissor])

        # Bind pipeline
        vk.vkCmdBindPipeline(commandBuffer=self._vk["graphic_cmd_buffer"],
                             pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeline=self._vk["graphic_pipeline"])

        # Bind vertex buffer object
        vk.vkCmdBindVertexBuffers(commandBuffer=self._vk["graphic_cmd_buffer"],
                                  firstBinding=0,
                                  bindingCount=1,
                                  pBuffers=[self._vk["vbo_vert_buffer"]],
                                  pOffsets=[0])
        
        # Bind uniform buffer descriptor set
        vk.vkCmdBindDescriptorSets(commandBuffer=self._vk["graphic_cmd_buffer"],
                                   pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   layout=self._vk["graphic_layout"],
                                   firstSet=0,
                                   descriptorSetCount=1,
                                   pDescriptorSets=[self._vk["graphic_set"]],
                                   dynamicOffsetCount=0,
                                   pDynamicOffsets=None)
        
        # Draw triangle
        vk.vkCmdDraw(commandBuffer=self._vk["graphic_cmd_buffer"], vertexCount=3, 
                     instanceCount=1, firstVertex=0, firstInstance=0)

        # End render pass and command buffer
        vk.vkCmdEndRenderPass(commandBuffer=self._vk["graphic_cmd_buffer"])
        vk.vkEndCommandBuffer(commandBuffer=self._vk["graphic_cmd_buffer"])

    def _recordComputeRgbaToRgbCommands(self):
        # Prepare data for recording command buffers
        command_begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        
        # Record command buffer
        vk.vkBeginCommandBuffer(commandBuffer=self._vk["compute_cmd_buffer"], pBeginInfo=command_begin_info)
        
        # Bind pipeline
        vk.vkCmdBindPipeline(commandBuffer=self._vk["compute_cmd_buffer"],
                             pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline=self._vk["compute_pipeline"])

        # Bind uniform buffer descriptor set
        vk.vkCmdBindDescriptorSets(commandBuffer=self._vk["compute_cmd_buffer"],
                                   pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   layout=self._vk["compute_layout"],
                                   firstSet=0,
                                   descriptorSetCount=1,
                                   pDescriptorSets=[self._vk["compute_set"]],
                                   dynamicOffsetCount=0,
                                   pDynamicOffsets=None)
                                   
        subresource_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        self._insertImageMemoryBarrier(self._vk["compute_cmd_buffer"], self._vk["color_attachment_image"], 0, 0,
                                       vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.VK_IMAGE_LAYOUT_GENERAL,
                                       vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       subresource_range)
        self._insertImageMemoryBarrier(self._vk["compute_cmd_buffer"], self._vk["compute_rgb_image"], 0, 0,
                                       vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL,
                                       vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       subresource_range)

        # Dispatch compute pipeline
        vk.vkCmdDispatch(commandBuffer=self._vk["compute_cmd_buffer"],
                         groupCountX=math.ceil(self._width / 16),
                         groupCountY=math.ceil(self._height / 16),
                         groupCountZ=1)
        
        # End render pass and command buffer
        vk.vkEndCommandBuffer(commandBuffer=self._vk["compute_cmd_buffer"])

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

    def _getSupportedDepthFormat(self):
        # Find highest precision packed format that is supported by device
        possible_formats = [
            vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
            vk.VK_FORMAT_D32_SFLOAT,
            vk.VK_FORMAT_D24_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM_S8_UINT,
            vk.VK_FORMAT_D16_UNORM
        ]
        for fmt in possible_formats:
            format_props = vk.vkGetPhysicalDeviceFormatProperties(physicalDevice=self._vk["physical_device"], format=fmt)
            if format_props.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return fmt
        
        print("Vk: WARNING> No suitable depth format supported")
        return 0

    def _findMemoryTypeIndex(self, supported_memory_indices, req_properties):
        device_mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice=self._vk["physical_device"])
        for i in range(device_mem_props.memoryTypeCount):
            supported = supported_memory_indices & (1 << i)
            sufficient = (device_mem_props.memoryTypes[i].propertyFlags & req_properties) == req_properties
            if supported and sufficient:
                return i
        return -1

    def _createImageView(self, width, height, img_format, img_usage, aspect_mask):
        # Create specified attachment image
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=img_format,
            extent=[width, height, 1], # vk.VkExtent3D(width=width, height=height, depth=1),
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

    def _insertBufferMemoryBarrier(self, command_buffer, buffer, src_access_mask, dst_access_mask,
                                   src_stage_mask, dst_stage_mask, offset, size):
        memory_barrier = vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            srcAccessMask=src_access_mask,
            dstAccessMask=dst_access_mask,
            offset=offset,
            size=size,
            buffer=buffer
        )
        vk.vkCmdPipelineBarrier(commandBuffer=command_buffer,
				                srcStageMask=src_stage_mask,
				                dstStageMask=dst_stage_mask,
				                dependencyFlags=0,
				                memoryBarrierCount=0,
                                pMemoryBarriers=[],
				                bufferMemoryBarrierCount=1,
                                pBufferMemoryBarriers=[memory_barrier],
                                imageMemoryBarrierCount=0,
                                pImageMemoryBarriers=[])

    def _copyFramebufferToRgb(self):
        buffer_size = self._width * self._height * 3
    
        # Make class member var (only allocate once)
        ubo_data = np.empty(2, dtype=np.uint32)
        ubo_data[0] = self._width
        ubo_data[1] = self._height
        ffi.memmove(self._vk["compute_uniform_memory_loc"], ubo_data, ubo_data.nbytes)
        
        # Submit compute command to graphics/compute queue
        self._submitWork(self._vk["compute_cmd_buffer"], self._vk["graphic_compute_queue"])
        
        # Create destination buffer to copy image into
        rgb_buffer = self._createBuffer(buffer_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Copy rendered image to host visible destination buffer
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
        #self._insertBufferMemoryBarrier(copy_command_buffer, rgb_buffer["buffer"], 0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        #                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, buffer_size)

        self._insertImageMemoryBarrier(copy_command_buffer, self._vk["compute_rgb_image"], 0, 0,
                                       vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       subresource_range)

        image_copy_region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0, # 0 => tightly packed... could also explicitly say `self._width`
            bufferImageHeight=0, # 0 => tightly packed... could also explicitly say `self._height`
            imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self._width * 3, height=self._height, depth=1)
        )
        vk.vkCmdCopyImageToBuffer(copy_command_buffer, srcImage=self._vk["compute_rgb_image"],
                                  srcImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  dstBuffer=rgb_buffer["buffer"],
                                  regionCount=1,
                                  pRegions=[image_copy_region])

        #self._insertBufferMemoryBarrier(copy_command_buffer, rgb_buffer["buffer"], vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        #                                vk.VK_ACCESS_MEMORY_READ_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        #                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, buffer_size)

        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)

        self._submitWork(copy_command_buffer, self._vk["graphic_compute_queue"])

        # Map image buffer memory so we can read it
        memory_location = vk.vkMapMemory(device=self._vk["device"],
                                         memory=rgb_buffer["memory"],
                                         offset=0,
                                         size=buffer_size,
                                         flags=0)

        # Convert memory to numpy array
        rgb = np.frombuffer(memory_location, dtype=np.uint8, count=buffer_size, offset=0)

        # Unmap buffer memory
        vk.vkUnmapMemory(device=self._vk["device"], memory=rgb_buffer["memory"])
        
        # Clean up
        vk.vkDestroyBuffer(device=self._vk["device"], buffer=rgb_buffer["buffer"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=rgb_buffer["memory"], pAllocator=None)
        vk.vkFreeCommandBuffers(device=self._vk["device"],
                                commandPool=self._vk["command_pool"],
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])

        return rgb

    def _copyFramebufferToRgba(self):
        buffer_size = self._width * self._height * 4

        # Create destination buffer to copy image into
        rgba_buffer = self._createBuffer(buffer_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Copy rendered image to host visible destination buffer
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
        self._insertBufferMemoryBarrier(copy_command_buffer, rgba_buffer["buffer"], 0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, buffer_size)

        image_copy_region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0, # 0 => tightly packed... could also explicitly say `self._width`
            bufferImageHeight=0, # 0 => tightly packed... could also explicitly say `self._height`
            imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self._width, height=self._height, depth=1)
        )
        vk.vkCmdCopyImageToBuffer(copy_command_buffer, srcImage=self._vk["color_attachment_image"],
                                  srcImageLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  dstBuffer=rgba_buffer["buffer"],
                                  regionCount=1,
                                  pRegions=[image_copy_region])

        self._insertBufferMemoryBarrier(copy_command_buffer, rgba_buffer["buffer"], vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                        vk.VK_ACCESS_MEMORY_READ_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, buffer_size)

        vk.vkEndCommandBuffer(commandBuffer=copy_command_buffer)

        self._submitWork(copy_command_buffer, self._vk["graphic_compute_queue"])

        # Map image buffer memory so we can read it
        memory_location = vk.vkMapMemory(device=self._vk["device"],
                                         memory=rgba_buffer["memory"],
                                         offset=0,
                                         size=buffer_size,
                                         flags=0)

        # Convert memory to numpy array
        rgba = np.frombuffer(memory_location, dtype=np.uint8, count=buffer_size, offset=0)

        # Unmap buffer memory
        vk.vkUnmapMemory(device=self._vk["device"], memory=rgba_buffer["memory"])
        
        # Clean up
        vk.vkDestroyBuffer(device=self._vk["device"], buffer=rgba_buffer["buffer"], pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=rgba_buffer["memory"], pAllocator=None)
        vk.vkFreeCommandBuffers(device=self._vk["device"],
                                commandPool=self._vk["command_pool"],
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])
        """
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
        
        self._submitWork(copy_command_buffer, self._vk["graphic_compute_queue"])
        
        # Vulkan: get layout of the image (including row pitch)
        subresource = vk.VkImageSubresource(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT)
        subresource_layout = vk.vkGetImageSubresourceLayout(device=self._vk["device"],
                                                            image=image,
                                                            pSubresource=[subresource])

        # Vulkan: map image memory so we can read it
        memory_location = vk.vkMapMemory(device=self._vk["device"],
                                         memory=image_memory,
                                         offset=0,
                                         size=subresource_layout.size,
                                         flags=0)
        
        # Convert memory to numpy array and extract RGB data
        rgba = None
        if subresource_layout.rowPitch == (self._width * 4):
            rgba = np.frombuffer(memory_location, dtype=np.uint8, count=subresource_layout.size, offset=subresource_layout.offset)
        else:
            rgba_raw = np.frombuffer(memory_location, dtype=np.uint8, count=subresource_layout.size, offset=subresource_layout.offset)
            rgba = np.empty(self._width * self._height * 4, dtype=np.uint8)
            out_pitch = self._width * 4
            for i in range(self._height):
                raw_start = i * subresource_layout.rowPitch
                out_start = i * out_pitch
                rgba[out_start:out_start + out_pitch] = rgba_raw[raw_start:raw_start + out_pitch]
        
        # Vulkan: unmap image memory
        vk.vkUnmapMemory(device=self._vk["device"], memory=image_memory)
        
        # Clean up
        vk.vkDestroyImage(device=self._vk["device"], image=image, pAllocator=None)
        vk.vkFreeMemory(device=self._vk["device"], memory=image_memory, pAllocator=None)
        vk.vkFreeCommandBuffers(device=self._vk["device"],
                                commandPool=self._vk["command_pool"],
                                commandBufferCount=1,
                                pCommandBuffers=[copy_command_buffer])
        """
        return rgba

    def _getRawImage(self):
        # Copy framebuffer to RGBA array
        #rgba = self._copyFramebufferToRgba()
        
        # Convert from RGBA to RGB
        #rgb_img = cv2.cvtColor(rgba.reshape((self._height, self._width, 4)), cv2.COLOR_RGBA2RGB).flatten()
        #rgb_img = np.delete(memory_array, np.arange(3, memory_array.size, 4))
        #rgb_img = memory_array.reshape(-1, 4)[:,:3].flatten()
        
        
        
        
        rgb_img = self._copyFramebufferToRgb()
        
        """
        # TEST -> save PPM image
        file = open("vk_image.ppm", "wb")
        file.write(f"P6\n{self._width} {self._height}\n255\n".encode("utf-8"))
        file.write(rgb_img.tobytes())
        file.close()
        """

        return rgb_img

    def _getJpegImage(self):
        # Copy framebuffer to RGBA array
        rgba = self._copyFramebufferToRgba()

        # Convert from RGBA to BGR
        bgr_img = cv2.cvtColor(rgba.reshape((self._height, self._width, 4)), cv2.COLOR_RGBA2BGR)
        
        result, encoded_img = cv2.imencode('.jpg', bgr_img, (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality))
        if result:
            return encoded_img
        return None

    def _distanceSqr2d(self, p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return (dx * dx) + (dy * dy)

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
