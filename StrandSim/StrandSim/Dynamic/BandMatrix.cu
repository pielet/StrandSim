#include "BandMatrix.h"
#include "../Utils/CUDAMathDef.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include "../Utils/EigenDef.h"

const double tol = 1e-12;

template <typename ScalarT>
__global__ void addInDiagonalKernal (MatrixWrapper<ScalarT>* matrix, ScalarT* m)
{
	int i = threadIdx.x;

	matrix->addInPlace(i, i, m[i]);
}

template <typename ScalarT>
CUDA_CALLABLE_MEMBER void MatrixWrapper<ScalarT>::addInPlace(int x, int y, ScalarT value)
{
	ScalarT* value_ptr = m_value + m_rowPtr[x] + (y - m_colIdx[m_rowPtr[x]]);
	*value_ptr += value;
}

template <typename ScalarT, int w>
BandMatrix<ScalarT, w>::BandMatrix(int n, cudaStream_t stream) :
	m_n(n),
	m_stream(stream)
{
	// Create context
	cusolverSpCreate(&m_handle);
	cusolverSpSetStream(m_handle, stream);
	cusparseCreateMatDescr(&m_descr);
	cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO);

	// Allocate memory
	m_nnz = 2 * (w - 2) * (n - w) + w * w;

	cudaMalloc((void**)& m_value, m_nnz * sizeof(ScalarT));
	cudaMalloc((void**)& m_rowPtr, (n + 1) * sizeof(int));
	cudaMalloc((void**)& m_colIdx, m_nnz * sizeof(int));
	cudaMalloc((void**)& m_matrix, sizeof(MatrixWrapper<ScalarT>));

	// Set value to 0
	setZero();

	// Compute rowPtr and colIdx
	std::vector<int> rowPtr(n + 1);
	std::vector<int> colIdx(m_nnz);

	int numVertices = (n + 1) / 4;
	int start = -(w - 3);
	int end = w;
	int count = 0;
	int start_k, end_k;
	for (int i = 0; i < numVertices; ++i)
	{
		start_k = std::max(0, start);
		end_k = std::min(n, end);
		for (int j = 0; j < 3; ++j)		// Vertices
		{
			rowPtr[4 * i + j] = count;
			for (int k = start_k; k < end_k; ++k)
			{
				colIdx[count++] = k;
			}
		}
		if (i < numVertices - 1)		// Thetas
		{
			start_k = std::max(0, start + 4);
			rowPtr[4 * i + 3] = count;
			for (int k = start_k; k < end_k; ++k)
			{
				colIdx[count++] = k;
			}
		}
		start += 4;
		end += 4;
	}
	rowPtr[n] = m_nnz;

	// Copy index data to device
	cudaMemcpyAsync(m_rowPtr, rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_colIdx, colIdx.data(), m_nnz * sizeof(int), cudaMemcpyHostToDevice, m_stream);
	
	MatrixWrapper<ScalarT> matrix(m_value, m_rowPtr, m_colIdx);
	cudaMemcpyAsync(m_matrix, &matrix, sizeof(matrix), cudaMemcpyHostToDevice, m_stream);
}

template <typename ScalarT, int w>
BandMatrix<ScalarT, w>::~BandMatrix()
{
	cudaFree(m_value);
	cudaFree(m_rowPtr);
	cudaFree(m_colIdx);

	cusolverSpDestroy(m_handle);
	cusparseDestroyMatDescr(m_descr);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::multiplyInPlace(ScalarT t)
{
	int block = 32;
	int grid = (m_nnz + block - 1) / block;
	multiply <<< grid, block, 0, m_stream >>> (m_nnz, m_value, t);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::addInDiagonal(ScalarT* m)
{
	addInDiagonalKernal <<< 1, m_n, 0, m_stream >>> (m_matrix, m);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::setZero()
{
	int block = 32;
	int grid = (m_nnz + block - 1) / block;
	set<ScalarT> <<< grid, block, 0, m_stream >>> (m_nnz, m_value, 0.0);
}

/* template function specialisation *********************************************/
template <>
bool CUDASolveChol<double>(BandMatrix<double, 11>* A, const double* b, double* x)
{
	int singularity;
	cusolverSpDcsrlsvchol(
		A->getHandle(), A->getn(), A->getnnz(),
		A->getDescr(), A->getValue(), A->getRowPtr(), A->getColIdx(),
		b, tol, 2, x, &singularity);  // symamd
	//Eigen::VecXx value(A->getnnz());
	//Eigen::VecXx res(A->getn());
	//Eigen::VecXx B(A->getn());
	//cudaMemcpy(value.data(), A->getValue(), A->getnnz() * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(res.data(), x, A->getn() * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(B.data(), b, A->getn() * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize(); 
	//std::cout << "A:\n" << value << "\nb:\n" << B << "\nx:\n" << res << std::endl;
	return singularity < 0;
}

template <>
bool CUDASolveChol<float>(BandMatrix<float, 11> * A, const float* b, float* x)
{
	int singularity;
	cusolverSpScsrlsvchol(
		A->getHandle(), A->getn(), A->getnnz(),
		A->getDescr(), A->getValue(), A->getRowPtr(), A->getColIdx(),
		b, tol, 2, x, &singularity);  // symamd 
	return singularity < 0;
}

template class MatrixWrapper<Scalar>;
template class BandMatrix<Scalar, 7>;
template class BandMatrix<Scalar, 11>;