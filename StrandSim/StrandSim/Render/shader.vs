#version 330 core
layout (location = 0) in vec3 aPos;

out vec4 frag_color;

uniform mat4 MVP;
uniform vec4 vertex_color;

void main()
{
	frag_color = vertex_color;
	gl_Position = MVP * vec4(aPos, 1.0);
}