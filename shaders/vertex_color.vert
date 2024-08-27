#version 450

layout(push_constant) uniform UniformData {
    mat4 mvp_matrix;
};

layout(location=0) in vec2 vertex_position;
layout(location=1) in vec3 vertex_color;

layout(location=0) out vec3 v_color;

void main() {
    gl_Position = mvp_matrix * vec4(vertex_position, 0.0, 1.0);
    v_color = vertex_color;
}
