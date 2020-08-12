#include "../Control/Parameters.h"

template <typename T, int n> class Vec;
typedef Vec<float, 3> Vec3f;
typedef Vec<Scalar, 3> Vec3x;
typedef Vec<Scalar, 4> Vec4x;
typedef Vec<Scalar, 11> Vec11x;

template <typename T, int n, int m> class Mat;
typedef Mat<Scalar, 3, 3> Mat3x;
typedef Mat<Scalar, 11, 11> Mat11x;
typedef Mat<Scalar, 11, 4> Mat11x4x;