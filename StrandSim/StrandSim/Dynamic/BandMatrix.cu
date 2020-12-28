#include "BandMatrix.h"
#include <vector>
#include <algorithm>
#include "../Utils/CUDAMathDef.h"

template <typename ScalarT>
CUDA_CALLABLE_MEMBER void MatrixWrapper<ScalarT>::add(int x, int y, ScalarT value)
{
	ScalarT* value_ptr = m_value + m_rowPtr[x] + (y - m_colIdx[m_rowPtr[x]]);
	atomicAdd(value_ptr, value);
}

//template <typename ScalarT>
//CUDA_CALLABLE_MEMBER void MatrixWrapper<ScalarT>::add(int x, int y, ScalarT value)
//{
//	ScalarT* value_ptr = m_value + m_rowPtr[x] + (y - m_colIdx[m_rowPtr[x]]);
//	*value_ptr += value;
//}

template <typename ScalarT>
CUDA_CALLABLE_MEMBER ScalarT MatrixWrapper<ScalarT>::get(int x, int y) const
{
	if (y < m_colIdx[m_rowPtr[x]] || y > m_colIdx[m_rowPtr[x + 1] - 1])
		return 0.;
	else
		return m_value[m_rowPtr[x] + (y - m_colIdx[m_rowPtr[x]])];
}

template <typename ScalarT>
__global__ void addInDiagonalKernel(int n, MatrixWrapper<ScalarT>* matrix, ScalarT* M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;

	matrix->add(i, i, M[i]);
}

template <typename ScalarT>
__global__ void addInDiagonalKernel(int w, const int* strand_ptr, ScalarT* A, const ScalarT* M)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;
	int n_dof = 4 * (strand_ptr[sid + 1] - strand_ptr[sid]) - 1;
	int start_dof = 4 * strand_ptr[sid] - sid;
	if (tid >= n_dof) return;

	A[start_dof * (2 * w + 1) + w * n_dof + tid] += M[start_dof + tid];
}

template <typename ScalarT>
__global__ void invDiagonalKernel(int n, MatrixWrapper<ScalarT>* matrix, ScalarT* diag)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;

	diag[i] = 1. / matrix->get(i, i);
}

template <typename ScalarT, int w>
__global__ void convertKernel(int n, const int* strand_ptr, MatrixWrapper<ScalarT>* matrix, ScalarT* band_values)
{
	int sid = blockDim.x * blockIdx.x + threadIdx.x;
	if (sid >= n) return;

	int n_ver = strand_ptr[sid + 1] - strand_ptr[sid];
	int n_dof = 4 * n_ver - 1;
	int start_dof = 4 * strand_ptr[sid] - sid;

	for (int i = 0; i < n_dof; ++i) {
		for (int j = 0; j < n_dof; ++j) {
			int band_idx = j - i + w / 2;
			if (band_idx >= 0 && band_idx < w) {
				band_values[start_dof * w + band_idx * n_dof + i] = matrix->get(start_dof + i, start_dof + j);
			}
		}
	}
}

template <typename ScalarT, int w>
BandMatrix<ScalarT, w>::BandMatrix(int num_dofs, int num_strands, const std::vector<int>& strand_ptr, const int* dev_strand_ptr):
	m_ns(num_strands),
	m_n(num_dofs),
	m_nnz(0),
	m_blocksPerGrid((num_dofs + g_threadsPerBlock - 1) / g_threadsPerBlock),
	m_strand_ptr(dev_strand_ptr)
{
	checkCudaErrors(cublasCreate(&m_cublasHandle));
	checkCudaErrors(cublasCreate(&m_bandValueHandle));

	// Allocate memory
	m_nnz = 2 * (w - 2) * (num_dofs - num_strands * w) + num_strands * w * w;

	checkCudaErrors(cudaMalloc((void**)& m_value, m_nnz * sizeof(ScalarT)));
	checkCudaErrors(cudaMalloc((void**)& m_rowPtr, (m_n + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_colIdx, m_nnz * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_matrix, sizeof(MatrixWrapper<ScalarT>)));

	checkCudaErrors(cudaMalloc((void**)&m_bandValues, m_n * (2 * w - 1) * sizeof(ScalarT)));
	cudaMemset(m_bandValues, 0, m_n * (2 * w - 1) * sizeof(ScalarT));

	// Set value to 0
	setZero();

	// Compute rowPtr and colIdx
	std::vector<int> rowPtr(m_n + 1);
	std::vector<int> colIdx(m_nnz);

	int num_vtx, num_dof;
	int start, end, start_k, end_k;
	int count = 0, accum_dof = 0;
	for (int si = 0; si < num_strands; ++si)
	{
		num_vtx = strand_ptr[si + 1] - strand_ptr[si];
		num_dof = 4 * num_vtx - 1;
		start = -(w - 3);
		end = w;
		for (int i = 0; i < num_vtx; ++i)
		{
			start_k = std::max(0, start) + accum_dof;
			end_k = std::min(num_dof, end) + accum_dof;
			for (int j = 0; j < 3; ++j)		// Vertices
			{
				rowPtr[accum_dof + 4 * i + j] = count;
				for (int k = start_k; k < end_k; ++k)
				{
					colIdx[count++] = k;
				}
			}
			if (i < num_vtx - 1)		// Thetas
			{
				start_k = std::max(0, start + 4) + accum_dof;
				rowPtr[accum_dof + 4 * i + 3] = count;
				for (int k = start_k; k < end_k; ++k)
				{
					colIdx[count++] = k;
				}
			}
			start += 4;
			end += 4;
		}
		accum_dof += num_dof;
	}
	rowPtr[m_n] = m_nnz;

	// Copy index data to device
	cudaMemcpy(m_rowPtr, rowPtr.data(), (m_n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_colIdx, colIdx.data(), m_nnz * sizeof(int), cudaMemcpyHostToDevice);
	
	MatrixWrapper<ScalarT> matrix(m_value, m_rowPtr, m_colIdx);
	cudaMemcpy(m_matrix, &matrix, sizeof(matrix), cudaMemcpyHostToDevice);
}

template <typename ScalarT, int w>
BandMatrix<ScalarT, w>::~BandMatrix()
{
	cublasDestroy(m_cublasHandle);

	cudaFree(m_value);
	cudaFree(m_rowPtr);
	cudaFree(m_colIdx);
	cudaFree(m_matrix);

	cublasDestroy(m_bandValueHandle);
	cudaFree(m_bandValues);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::convertToBand()
{
	cudaMemset(m_bandValues, 0, m_n * (2 * w - 1) * sizeof(ScalarT));
	convertKernel<ScalarT, 2 * w - 1> <<< (m_ns + g_threadsPerBlock - 1) / g_threadsPerBlock, g_threadsPerBlock >>> (m_ns, m_strand_ptr, m_matrix, m_bandValues);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::scale(ScalarT t)
{
	CublasCaller<ScalarT>::scal(m_cublasHandle, m_nnz, &t, m_value);
	CublasCaller<ScalarT>::scal(m_bandValueHandle, m_n * (2 * w - 1), &t, m_bandValues);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::addInDiagonal(ScalarT* M)
{
	addInDiagonalKernel <<< m_blocksPerGrid, g_threadsPerBlock >>> (m_n, m_matrix, M);
	addInDiagonalKernel <<< m_ns, (4 * g_maxVertex - 1 + g_threadsPerBlock - 1) / g_threadsPerBlock * g_threadsPerBlock >>> (10, m_strand_ptr, m_bandValues, M);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::invDiagonal(ScalarT* diag) const
{
	invDiagonalKernel <<< m_blocksPerGrid, g_threadsPerBlock >>> (m_n, m_matrix, diag);
}

template <typename ScalarT, int w>
void BandMatrix<ScalarT, w>::setZero()
{
	cudaMemset(m_value, 0, m_nnz * sizeof(ScalarT));
	cudaMemset(m_bandValues, 0, m_n * (2 * w - 1) * sizeof(ScalarT));
}

template class MatrixWrapper<Scalar>;
template class BandMatrix<Scalar, 7>;
template class BandMatrix<Scalar, 11>;