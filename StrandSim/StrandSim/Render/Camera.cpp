#include "Camera.h"
#include "../Utils/EigenDef.h"

const float mouse_speed = 1.f;
const float wheel_speed = 0.01f;

float degToRad(float deg)
{
	return deg / 180 * M_PI;
}

Eigen::Mat4f perspective(float fovy, float aspect, float zNear, float zFar)
{
	assert(aspect > 0);
	assert(zFar > zNear);

	float radf = degToRad(fovy);

	float tanHalfFovy = tan(radf / 2.0);
	Eigen::Mat4f res = Eigen::Mat4f::Zero();
	res(0, 0) = 1.0f / (aspect * tanHalfFovy);
	res(1, 1) = 1.0f / (tanHalfFovy);
	res(2, 2) = -(zFar + zNear) / (zFar - zNear);
	res(3, 2) = -1.0f;
	res(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);

	return res;
}

Eigen::Mat4f lookAt(const Eigen::Vec3f& eye, const Eigen::Vec3f& center, const Eigen::Vec3f& up)
{
	Eigen::Vec3f f = (center - eye).normalized();
	Eigen::Vec3f u = up.normalized();
	Eigen::Vec3f s = f.cross(u).normalized();
	u = s.cross(f);

	Eigen::Mat4f res;
	res << s.x(),  s.y(),  s.z(), -s.dot(eye),
		   u.x(),  u.y(),  u.z(), -u.dot(eye),
		  -f.x(), -f.y(), -f.z(),  f.dot(eye),
		  0, 0, 0, 1;

	return res;
}

Camera::Camera(int w, int h):
	m_viewPort(w, h),
	m_zClipping(0.01, 100)
{
	setPose(Eigen::Vec3f(0.0f, 0.0f, 5.0f), Eigen::Vec3f::Zero());
}

void Camera::setPose(const Eigen::Vec3f& eye, const Eigen::Vec3f& lookAt)
{
	m_eye = eye;
	m_lookAt = lookAt;
	Eigen::Vec3f dir = (lookAt - eye).normalized();
	m_right = dir.cross(Eigen::Vec3f(0, 1, 0)).normalized();
	m_up = m_right.cross(dir);
}

void Camera::setViewPort(int w, int h)
{
	m_viewPort = Eigen::Vec2i(w, h);
}

void Camera::setZClipping(float near, float far)
{
	m_zClipping = Eigen::Vec2f(near, far);
}

void Camera::beginMotion(Motion motion, int x, int y)
{
	m_motion = motion;
	m_x = x;
	m_y = y;
}

void Camera::move(int x, int y)
{
	float alpha = mouse_speed * ((m_lookAt - m_eye).norm() + 1.f);
	float dx = float(x - m_x) / m_viewPort(0);
	float dy = float(y - m_y) / m_viewPort(1);

	switch (m_motion)
	{
	case Camera::SCALE:
	{
		Eigen::Vec3f dir = m_up.cross(m_right).normalized();
		m_eye -= alpha * dy * dir;
		break;
	}
	case Camera::ROTATE:
	{
		float x_angle = mouse_speed * dx;
		float y_angle = mouse_speed * dy;
		Eigen::Vec3f new_dir = Eigen::AngleAxis<float>(-x_angle, m_up) * Eigen::AngleAxis<float>(-y_angle, m_right) * (m_eye - m_lookAt);
		m_eye = m_lookAt + new_dir;
		m_right = Eigen::AngleAxis<float>(-x_angle, m_up) * m_right;
		m_up = m_right.cross(-new_dir).normalized();
		break;
	}
	case Camera::TRANSLATE:
	{
		Eigen::Vec3f delta_p = dx * m_right - dy * m_up;
		m_eye -= alpha * delta_p;
		m_lookAt -= alpha * delta_p;
		break;
	}
	}

	m_x = x;
	m_y = y;
}

void Camera::scroll(int direction)
{
	float alpha = wheel_speed * ((m_lookAt - m_eye).norm() + 1.f);
	m_eye += direction * alpha * m_up.cross(m_right).normalized();
}

Eigen::Mat4f Camera::getViewMatrix() const
{
	return lookAt(m_eye, m_lookAt, m_up);
}

Eigen::Mat4f Camera::getPerspectiveMatrix() const
{
	return perspective(60.f, float(m_viewPort(0)) / m_viewPort(1), m_zClipping(0), m_zClipping(1));
}
