#version 450

layout(std140, binding=0) uniform UniformData {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location=0) in vec2 vertex_position;
layout(location=1) in vec3 vertex_color;

layout(location=0) out vec3 v_color;

void main() {
    // Transform/project vertex
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(vertex_position, 0.0, 1.0);
    
    // Correct depth (Vulkan [0,1], whereas OpenGL [-1,1])
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
    
    // Forward vetex color to fragment shader
    v_color = vertex_color;
}
