#ifndef CUBLAS_H
#define CUBLAS_H

#include <cublas_v2.h>
#include "helper_cuda.h"

template <typename T> struct CublasCaller;

template<>
struct CublasCaller<double>
{
	static void copy(cublasHandle_t handle, int n, const double* x, double* y)
	{
		checkCudaErrors(cublasDcopy(handle, n, x, 1, y, 1));
	}

	static void dot(cublasHandle_t handle, int n, const double* x, const double* y, double* res)
	{
		checkCudaErrors(cublasDdot(handle, n, x, 1, y, 1, res));
	}

	static void nrm2(cublasHandle_t handle, int n, const double* x, double* res)
	{
		checkCudaErrors(cublasDnrm2(handle, n, x, 1, res));
	}

	static void scal(cublasHandle_t handle, int n, const double* alpha, double* x, int strip = 1)
	{
		checkCudaErrors(cublasDscal(handle, n, alpha, x, strip));
	}

	static void axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, double* y)
	{
		checkCudaErrors(cublasDaxpy(handle, n, alpha, x, 1, y, 1));
	}

	static void sum(cublasHandle_t handle, int n, const double* x, double* res)
	{
		checkCudaErrors(cublasDasum(handle, n, x, 1, res));
	}

	static void sbmv(cublasHandle_t handle, int n, const double* alpha, const double* A, const double* x, const double* beta, double* y)
	{
		checkCudaErrors(cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, n, 0, alpha, A, 1, x, 1, beta, y, 1));
	}
};

template<>
struct CublasCaller<float>
{
	static void copy(cublasHandle_t handle, int n, const float* x, float* y)
	{
		checkCudaErrors(cublasScopy(handle, n, x, 1, y, 1));
	}

	static void dot(cublasHandle_t handle, int n, const float* x, const float* y, float* res)
	{
		checkCudaErrors(cublasSdot(handle, n, x, 1, y, 1, res));
	}

	static void scal(cublasHandle_t handle, int n, const float* alpha, float* x, int strip = 1)
	{
		checkCudaErrors(cublasSscal(handle, n, alpha, x, strip));
	}

	static void axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, float* y)
	{
		checkCudaErrors(cublasSaxpy(handle, n, alpha, x, 1, y, 1));
	}

	static void sum(cublasHandle_t handle, int n, const float* x, float* res)
	{
		checkCudaErrors(cublasSasum(handle, n, x, 1, res));
	}
};

#endif // !CUBLAS_H
