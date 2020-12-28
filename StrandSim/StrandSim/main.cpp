// OpenGL
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <iostream>

#include "Render/Camera.h"
#include "Render/Shader.h"
#include "Utils/Timer.h"
#include "Control/ProblemManager.h"

// parameters
int g_window_width = 800;
int g_window_height = 600;

Timer g_timer;
int frame_count = 0;

Shader* g_shader;
Camera* g_camera;
ProblemManager* g_problem;

bool g_stop = true;
bool g_step = false;

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

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q': case 'Q':
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
		glutLeaveMainLoop();
		break;
	case ' ':
		g_stop = !g_stop;
		break;
	case 's': case 'S':
		g_step = true;
		break;
	}
}

void display()
{
	glClearColor(1.f, 1.f, 1.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	g_shader->use();
	Eigen::Mat4f MVP = g_camera->getPerspectiveMatrix() * g_camera->getViewMatrix();
	g_shader->setMat4f("MVP", MVP);

	if (g_stop)
	{
		if (g_step)
		{
			g_problem->step();
			g_step = false;
		}
	}
	else g_problem->step();

	g_problem->render(g_shader);

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
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, g_window_width, g_window_height);

	// glut callback
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutMouseWheelFunc(wheelScroll);

	// scene
	if (std::string(typeid(Scalar).name()) == std::string("double"))
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	g_shader = new Shader("Render/shader.vs", "Render/shader.fs");
	g_camera = new Camera(g_window_width, g_window_height);
	g_problem = new ProblemManager("../Assets/test.xml");

	glutMainLoop();

	delete g_shader;
	delete g_camera;
	delete g_problem;

	cudaDeviceReset();

	std::cout << "EXIT.";

	return 0;
}
