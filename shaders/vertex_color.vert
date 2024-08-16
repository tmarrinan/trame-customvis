#version 450

layout(set=0,binding=0) uniform UniformData {
    mat4 mvp_matrix;
} uniform_data;

layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_color;

layout(location=0) out vec3 v_color;

void main() {
    gl_Position = uniform_data.mvp_matrix * vec4(vertex_position, 1.0);
    v_color = vertex_color;
}
