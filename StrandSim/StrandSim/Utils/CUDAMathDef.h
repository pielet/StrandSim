#ifndef VECTORS_H
#define VECTORS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include "../Control/Parameters.h"

template <typename T> __device__ T SQRT(const T& x);
template <> __device__ __forceinline__ double SQRT(const double& x)
{
	return sqrt(x);
}
template <> __device__ __forceinline__ float SQRT(const float& x)
{
	return sqrtf(x);
}

template <typename T> __device__ T FAB(const T& x);
template <> __device__ __forceinline__ double FAB(const double& x)
{
	return fabs(x);
}
template <> __device__ __forceinline__ float FAB(const float& x)
{
	return fabsf(x);
}

template <typename T> __device__ T MAX(const T& x, const T& y);
template <> __device__ __forceinline__ double MAX(const double& x, const double& y)
{
	return fmax(x, y);
}
template <> __device__ __forceinline__ float MAX(const float& x, const float& y)
{
	return fmaxf(x, y);
}


template <typename T, int n>
struct Vec
{
	T value[n];
	__host__ __device__ Vec()
	{
		for (int i = 0; i < n; ++i) value[i] = 0.;
	}
	__host__ __device__ Vec(T x, T y, T z)
	{
		if constexpr (n == 3)
		{
			value[0] = x;
			value[1] = y;
			value[2] = z;
		}
		else Vec();
	}
	__host__ __device__ Vec(T x, T y, T z, T w)
	{
		if constexpr (n == 4)
		{
			value[0] = x;
			value[1] = y;
			value[2] = z;
			value[3] = w;
		}
		else Vec();
	}
	__host__ __device__ Vec(int size, const T* val)
	{
		assert(size == n);
		for (int i = 0; i < n; ++i) value[i] = val[i];
	}
	__host__ __device__ Vec(const Vec<T, n>& v)
	{
		for (int i = 0; i < n; ++i) value[i] = v(i);
	}
	__host__ __device__ Vec<T, n> operator=(const Vec<T, n>& v)
	{
		for (int i = 0; i < n; ++i) value[i] = v(i);
		return *this;
	}
	__device__ Vec<T, n> operator+=(const Vec<T, n>& v)
	{
		for (int i = 0; i < n; ++i) value[i] += v(i);
		return *this;
	}
	__host__ __device__ T& operator()(int i)
	{
		return value[i];
	}
	__host__ __device__ const T& operator()(int i) const
	{
		return value[i];
	}
	__device__ Vec<T, n> operator-() const
	{
		Vec<T, n> vec;
		for (int i = 0; i < n; ++i) vec(i) = -value[i];
		return vec;
	}
	__device__ Vec<T, n> operator+(const Vec<T, n>& a) const
	{
		Vec<T, n> vec;
		for (int i = 0; i < n; ++i) vec(i) = value[i] + a(i);
		return vec;
	}
	__device__ Vec<T, n> operator-(const Vec<T, n>& a) const
	{
		Vec<T, n> vec;
		for (int i = 0; i < n; ++i) vec(i) = value[i] - a(i);
		return vec;
	}
	__device__ Vec<T, n> operator*(const T& a) const
	{
		Vec<T, n> vec;
		for (int i = 0; i < n; ++i) vec(i) = a * value[i];
		return vec;
	}
	__device__ Vec<T, n> operator/(const T& a) const
	{
		Vec<T, n> vec;
		for (int i = 0; i < n; ++i) vec(i) = value[i] / a;
		return vec;
	}
	__device__ T dot(const Vec<T, n>& a) const
	{
		T res = 0.;
		for (int i = 0; i < n; ++i) res += value[i] * a(i);
		return res;
	}
	__device__ T norm2() const
	{
		return dot(*this);
	}
	__device__ T norm() const
	{
		return SQRT(norm2());
	}
	__device__ Vec<T, n> normalized() const
	{
		return *this / norm();
	}
	__device__ Vec<T, 3> cross(const Vec<T, 3> & a) const
	{
		Vec<T, 3> vec;
		if constexpr (n == 3)
		{
			vec(0) = value[1] * a(2) - value[2] * a(1);
			vec(1) = value[2] * a(0) - value[0] * a(2);
			vec(2) = value[0] * a(1) - value[1] * a(0);
		}
		return vec;
	}

	template <int m>
	__device__ void assignAt(int i, const Vec<T, m>& vec)
	{
		assert(i + m <= n);
		for (int j = 0; j < m; ++j) value[i + j] = vec(j);
	}

	template <int m>
	__device__ Vec<T, m> at(int i) const
	{
		assert(i + m <= n);
		Vec<T, m> vec;
		for (int j = 0; j < m; ++j) vec(j) = value[i + j];
		return vec;
	}

	static __device__ Vec<T, n> Zero()
	{
		return Vec<T, n>();
	}
};

template <typename T, int n>
__device__ Vec<T, n> operator*(T a, const Vec<T, n>& vec)
{
	Vec<T, n> res = vec;
	for (int i = 0; i < n; ++i) res(i) *= a;
	return res;
}


template <typename T, int n, int m>
struct Mat
{
	T value[n][m];
	__device__ Mat()
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
			value[i][j] = 0.;
		}
	}
	__device__ Mat(const Mat<T, n, m>& mat)
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
				value[i][j] = mat(i, j);
		}
	}
	__device__ Mat<T, n, m> operator=(const Mat<T, n, m>& mat)
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				value[i][j] = mat(i, j);
		}
		return *this;
	}
	__device__ const T& operator()(int i, int j) const
	{
		return value[i][j];
	}
	__device__ T& operator()(int i, int j)
	{
		return value[i][j];
	}
	__device__ Mat<T, n, m> operator+=(const Mat<T, n, m>& mat)
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				value[i][j] += mat(i, j);
		}
		return *this;
	}
	__device__ Mat<T, n, m> operator-() const
	{
		Mat<T, n, m> mat;
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				mat(i, j) = -value[i][j];
		}
		return mat;
	}
	__device__ Mat<T, n, m> operator+(const Mat<T, n, m>& A) const
	{
		Mat<T, n, m> mat;
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				mat(i, j) = value[i][j] + A(i, j);
		}
		return mat;
	}
	__device__ Mat<T, n, m> operator-(const Mat<T, n, m>& A) const
	{
		Mat<T, n, m> mat;
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				mat(i, j) = value[i][j] - A(i, j);
		}
		return mat;
	}
	__device__ Mat<T, n, m> operator*(T a) const
	{
		Mat<T, n, m> mat;
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				mat(i, j) = a * value[i][j];
		}
		return mat;
	}
	__device__ Mat<T, n, m> operator*=(T a)
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
			value[i][j] *= a;
		}
		return *this;
	}
	__device__ void setZero()
	{
		for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
				value[i][j] = 0.;
		}
	}
	__device__ Mat<T, m, n> transpose() const
	{
		Mat<T, m, n> mat;
		for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
				mat(i, j) = value[j][i];
		}
		return mat;
	}

	template <int nn, int mm>
	__device__ void assignAt(int si, int sj, const Mat<T, nn, mm>& mat)
	{
		assert((si + nn <= n) && (sj + mm <= m));
		for (int i = 0; i < nn; ++i) for (int j = 0; j < mm; ++j) {
			value[si + i][sj + j] = mat(i, j);
		}
	}

	template <bool vertical = true, int nn = 3>
	__device__ void assignAt(int si, int sj, const Vec<T, nn>& vec)
	{
		if constexpr (vertical)
		{
			assert((si + nn <= n) && (sj < m));
			for (int i = 0; i < nn; ++i) value[si + i][sj] = vec(i);
		}
		else
		{
			assert((si < n) && (sj + nn <= m));
			for (int j = 0; j < nn; ++j) value[si][sj + j] = vec(j);
		}
	}

	template <int nn>
	__device__ Vec<T, nn> at(int si, int sj) const
	{
		assert((si + nn <= n) && (sj < m));
		Vec<T, nn> vec;
		for (int i = 0; i < nn; ++i) vec(i) = value[si + i][sj];
		return vec;
	}

	static __device__ Mat<T, n, n> Identity()
	{
		assert(n == m);
		Mat<T, n, n> mat;
		for (int i = 0; i < n; ++i) mat(i, i) = 1.;
		return mat;
	}
};

template <typename T, int n, int m>
__device__ Mat<T, n, m> operator*(T a, const Mat<T, n, m>& A)
{
	Mat<T, n, m> mat;
	for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
			mat(i, j) = a * A(i, j);
	}
	return mat;
}

/* Mat & Vec function ***************************************************************************
 */

template <typename T, int n>
__device__ Mat<T, n, n> symPart(const Mat<T, n, n>& A)
{
	return 0.5 * (A + A.transpose());
}

template <typename T, int n>
__device__ Mat<T, n, n> outerProd(const Vec<T, n>& a)
{
	Mat<T, n, n> mat;
	for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
			mat(i, j) = mat(j, i) = a(i) * a(j);
	}
	return mat;
}

template <typename T, int n>
__device__ Mat<T, n, n> outerProd(const Vec<T, n>& a, const Vec<T, n>& b)
{
	Mat<T, n, n> mat;
	for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
		mat(i, j) = a(i) * b(j);
	}
	return mat;
}

template <typename T>
__device__ Mat<T, 3, 3> crossMat(const Vec<T, 3>& a)
{
	Mat<T, 3, 3> mat;
	mat(0, 0) = mat(1, 1) = mat(2, 2) = 0.;
	mat(2, 1) = a(0);
	mat(1, 2) = -a(0);
	mat(0, 2) = a(1);
	mat(2, 0) = -a(1); 
	mat(1, 0) = a(2);
	mat(0, 1) = -a(2);
	return mat;
}

template <typename T, int n, int m>
__device__ Vec<T, n> operator*(const Mat<T, n, m>& A, const Vec<T, m>& v)
{
	Vec<T, n> x;
	for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j)
		x(i) += A(i, j) * v(j);
	return x;
}

template <typename T, int n, int m, int l>
__device__ Mat<T, n, l> operator*(const Mat<T, n, m>& A, const Mat<T, m, l>& B)
{
	Mat<T, n, l> X;
	for (int i = 0; i < n; ++i) for (int j = 0; j < l; ++j) for(int k = 0;k < m;++k)
		X(i, j) += A(i, k) * B(k, j);
	return X;
}

template <typename T, int n, int m>
__device__ Mat<T, n, n> symProd(const Mat<T, n, m>& A)
{
	Mat<T, n, n> B;
	for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
		T vel = 0;
		for (int k = 0; k < m; ++k)
			vel += A(i, k) * A(j, k);
		B(i, j) = B(j, i) = vel;
	}
	return B;
}

template <typename T, int n>
__device__ bool isSymmetric(const Mat<T, n, n>& A)
{
	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) {
		if (FAB(A(i, j) - A(j, i)) > 1e-12) return false;
	}
	return true;
}

/* VecXx functions ******************************************************************************
 */

// Set every entry in vec to a
template <typename T>
__global__ void set(int size, T* vec, T a)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) vec[i] = a;
}

template <typename T, typename ScalarT = Scalar>
__global__ void assign(T* a, const T* b, ScalarT alpha = 1.0)
{
	int i = threadIdx.x;
	a[i] = alpha * b[i];
}

// vec *= t
template <typename T>
__global__ void multiply(int size, T* vec, T t)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) vec[i] *= t;
}

// Compute c = a + alpha * b
template <typename T>
__global__ void add(T* c, const T* a, const T* b, T alpha = 1.0)
{
	int i = threadIdx.x;
	c[i] = a[i] + alpha * b[i];
}

// res = a^T * M * b
template <typename T>
__global__ void dot(T* res, const T* a, const T* b, const T* m)
{
	int i = threadIdx.x;

	extern __shared__ T c[];
	c[i] = a[i] * m[i] * b[i];
	__syncthreads();

	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
			c[i] += c[i + strip];
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	res[i] = c[i];
}

template <typename T>
__global__ void dot(T* res, const T* a, const T* b)
{
	int i = threadIdx.x;

	extern __shared__ T c[];
	c[i] = a[i] * b[i];
	__syncthreads();

	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
			c[i] += c[i + strip];
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	res[i] = c[i];
}

/* Vector geometry function ***************************************************************
 */

/* Rotates v around z by theta * pi, assuming z is unit vector.
 */
template <typename T>
__device__ void rotateAxisAngle(Vec<T, 3>& v, const Vec<T, 3>& z, T theta)
{
	T c = cospi(theta);
	T s = sinpi(theta);

	v = c * v + s * z.cross(v) + z.dot(v) * (1 - c) * z;
}

/* Computes the signed angle from u to v. Rotation is measured around n
 */
template <typename T>
__device__ T signedAngle(const Vec<T, 3>& u, const Vec<T, 3>& v, const Vec<T, 3>& n)
{
	Vec<T, 3> w = u.cross(v);
	T angle = atan2(w.norm(), u.dot(v));
	if (n.dot(w) < 0)
		return -angle;
	return angle;
}

/* Parallel-transport u along t0->t1, assuming u is orthogonal to t0 and all are unit vectors
 */
template <typename T>
__device__ Vec<T, 3> orthonormalParallelTransport(const Vec<T, 3>& u, const Vec<T, 3>& t0, const Vec<T, 3>& t1)
{
	Vec<T, 3> b = t0.cross(t1);
	T bNorm = b.norm();
	if (bNorm < 1e-12)
		return u;
	b = b / bNorm;

	Vec<T, 3> n0 = t0.cross(b);
	Vec<T, 3> n1 = t1.cross(b);

	return u.dot(n0) * n1 + u.dot(b) * b;
}

#endif // !VECTORS_H
