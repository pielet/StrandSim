#ifndef DEFINITION_H
#define DEFINITION_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "../Control/Parameters.h"

namespace Eigen
{
	typedef Matrix<int, 2, 1> Vec2i;
	typedef Matrix<float, 2, 1> Vec2f;
	typedef Matrix<float, 3, 1> Vec3f;
	typedef Matrix<float, 4, 1> Vec4f;
	typedef Matrix<float, Eigen::Dynamic, 1> VecXf;
	typedef Matrix<float, 4, 4> Mat4f;

	typedef Matrix<Scalar, 2, 1> Vec2x; ///< 2d scalar vector
	typedef Matrix<Scalar, 3, 1> Vec3x; ///< 3d scalar vector
	typedef Matrix<Scalar, 4, 1> Vec4x; ///< 4d scalar vector
	typedef Matrix<Scalar, Eigen::Dynamic, 1> VecXx; ///< arbitrary dimension scalar vector

	typedef Matrix<Scalar, 2, 2> Mat2x; ///< 2x2 scalar matrix
	typedef Matrix<Scalar, 3, 3> Mat3x; ///< 3x3 scalar matrix
	typedef Matrix<Scalar, 4, 4> Mat4x; ///< 4x4 scalar matrix
	typedef Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatXx; ///< arbitrary dimension scalar matrix
}

inline Scalar square(Scalar x)
{
	return x * x;
}

inline Scalar cubic(Scalar x)
{
	return x * x * x;
}

inline Scalar biquad(Scalar x)
{
	return x * x * x * x;
}
#endif // !DEFINITION_H
