// OpenGL
#include <GL\glew.h>
#include <GL\freeglut.h>
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "Utils/Definition.h"
#include "Utils/Timer.h"
#include "Render/Camera.h"
#include "Render/Shader.h"

// parameters
int g_window_width = 800;
int g_window_height = 600;
const int g_num_particles = 10000;
const float g_dt = 0.1;//s

GLuint g_vbo;
GLuint g_vao;
struct cudaGraphicsResource* cuda_vbo_resource;

Timer g_timer;
int frame_count = 0;

Shader* g_shader;
Camera* g_camera;

void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_resource, unsigned int flag)
{
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initial position
	unsigned int size = g_num_particles * 2 * 3 * sizeof(float);
	Eigen::VecXf initial_state = Eigen::VecXf::Zero(g_num_particles * 6);
	for (int i = 0; i < g_num_particles; ++i) 
	{
		initial_state.segment<3>(6 * i + 3) = Eigen::Vec3f::Random();
	}
	glBufferData(GL_ARRAY_BUFFER, size, initial_state.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(vbo_resource, *vbo, flag);
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_resource)
{
	cudaGraphicsUnregisterResource(vbo_resource);
	glDeleteBuffers(1, vbo);
}

__global__ void update(float3* state, float dt)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float3& pos = state[idx * 2];
	float3& vel = state[idx * 2 + 1];

	pos.x += vel.x * dt;
	pos.y += vel.y * dt;
	pos.z += vel.z * dt;

	// check collision
	if (pos.x > 1.0 || pos.x < -1.0) {
		vel.x = -vel.x;
		if (pos.x > 1.0)
			pos.x = 2.0 - pos.x;
		else
			pos.x = -2.0 - pos.x;
	}
	if (pos.y > 1.0 || pos.y < -1.0) {
		vel.y = -vel.y;
		if (pos.y > 1.0)
			pos.y = 2.0 - pos.y;
		else
			pos.y = -2.0 - pos.y;
	}
	if (pos.z > 1.0 || pos.z < -1.0) {
		vel.z = -vel.z;
		if (pos.z > 1.0)
			pos.z = 2.0 - pos.z;
		else
			pos.z = -2.0 - pos.z;
	}
}

void runCuda(struct cudaGraphicsResource**vbo_resource)
{
	float3* dptr;
	size_t num_bytes;
	cudaGraphicsMapResources(1, vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

	// launch kernal
	dim3 block(8);
	dim3 grid((g_num_particles + (block.x - 1)) / block.x);
	update << <grid, block >> > (dptr, g_dt);

	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void computePFS()
{
	float dt = g_timer.elapsedSeconds();

	std::stringstream oss;
	oss << "StrandSim: " << frame_count / dt << " fps";
	glutSetWindowTitle(oss.str().c_str());

	frame_count = 0;
	g_timer.start();
}

void reshape(int w, int h)
{
	g_window_width = w;
	g_window_height = h;

	g_camera->setViewPort(w, h);
	glViewport(0, 0, w, h);

	glutPostRedisplay();
}

void mouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		switch (button)
		{
		case GLUT_LEFT_BUTTON:
			g_camera->beginMotion(Camera::Motion::ROTATE, x, y);
			break;
		case GLUT_MIDDLE_BUTTON:
			g_camera->beginMotion(Camera::Motion::TRANSLATE, x, y);
			break;
		case GLUT_RIGHT_BUTTON:
			g_camera->beginMotion(Camera::Motion::SCALE, x, y);
			break;
		default:
			break;
		}
	}
}

void mouseMotion(int x, int y)
{
	g_camera->move(x, y);
}

void wheelScroll(int wheel, int direction, int x, int y)
{
	g_camera->scroll(direction);
}

void display()
{
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1, 1, 1);
	
	g_shader->use();
	Eigen::Mat4f MPV = g_camera->getPerspectiveMatrix() * g_camera->getViewMatrix();
	g_shader->setMat4f("MVP", MPV);

	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
	glPointSize(2.0);
	glDrawArrays(GL_POINTS, 0, g_num_particles);

	glutSwapBuffers();

	++frame_count;
	if (g_timer.elapsedSeconds() > 1.0)
		computePFS();
}

int main(int argc, char** argv)
{
	// glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(g_window_width, g_window_height);
	glutCreateWindow("StrandSim");
	g_timer.start();

	// glew
	glewInit();
	glViewport(0, 0, g_window_width, g_window_height);
	
	// glut callback
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutMouseWheelFunc(wheelScroll);

	// scene
	g_shader = new Shader("Render/shader.vs", "Render/shader.fs");
	g_camera = new Camera(g_window_width, g_window_height);

	createVBO(&g_vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsNone);
	runCuda(&cuda_vbo_resource);

	glutMainLoop();

	delete g_shader;
	delete g_camera;
	deleteVBO(&g_vbo, cuda_vbo_resource);

	return 0;
}
