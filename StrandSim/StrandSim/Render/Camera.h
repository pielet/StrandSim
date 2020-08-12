#ifndef CAMERA_H
#define CAMERA_H

#include "../Utils/EigenDef.h"

class Camera
{
public:
	enum Motion {SCALE, ROTATE, TRANSLATE};

	Camera(int w, int h);
	~Camera() {};

	void setPose(const Eigen::Vec3f& eye, const Eigen::Vec3f& lookAt);
	void setViewPort(int w, int h);
	void setZClipping(float near, float far);

	void beginMotion(Motion motion, int x, int y);
	void move(int x, int y);
	void scroll(int direction);

	Eigen::Mat4f getViewMatrix() const;
	Eigen::Mat4f getPerspectiveMatrix() const;

protected:
	Eigen::Vec3f m_eye;
	Eigen::Vec3f m_lookAt;
	Eigen::Vec3f m_right;
	Eigen::Vec3f m_up;

	Motion m_motion;
	int m_x;
	int m_y;

	Eigen::Vec2i m_viewPort;
	Eigen::Vec2f m_zClipping;
};

#endif // !CAMERA_H
