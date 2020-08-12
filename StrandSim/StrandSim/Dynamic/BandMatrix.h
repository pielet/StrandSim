#ifndef BAND_MATRIX_H
#define BAND_MATRIX_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "../Control/Parameters.h"
#include "cusparse.h"
#include "cusolverSp.h"

template <typename ScalarT>
struct MatrixWrapper
{
	ScalarT* m_value;
	int* m_rowPtr;
	int* m_colIdx;
	MatrixWrapper(ScalarT* value, int* rowPtr, int* colIdx) :
		m_value(value), m_rowPtr(rowPtr), m_colIdx(colIdx) {}
	CUDA_CALLABLE_MEMBER void addInPlace(int x, int y, ScalarT value);
};

template <typename ScalarT, int w>
class BandMatrix
{
public:
	BandMatrix(int n, cudaStream_t stream);
	~BandMatrix();

	const ScalarT* getValue() const { return m_value; }
	const int* getRowPtr() const { return m_rowPtr; }
	const int* getColIdx() const { return m_colIdx; }
	MatrixWrapper<ScalarT>* getMatrix() { return m_matrix; }

	int getn() const { return m_n; }
	int getnnz() const { return m_nnz; }
	cusolverSpHandle_t getHandle() { return m_handle; }
	cusparseMatDescr_t getDescr() { return m_descr; }

	void multiplyInPlace(ScalarT t);
	void addInDiagonal(ScalarT* m);
	void setZero();

protected:
	int m_n;	// dof
	int m_nnz;	// non-zero
	cudaStream_t m_stream;
	cusolverSpHandle_t m_handle;
	cusparseMatDescr_t m_descr;

	ScalarT* m_value;
	int* m_rowPtr;
	int* m_colIdx;
	MatrixWrapper<ScalarT>* m_matrix;
};

template <typename ScalarT>
bool CUDASolveChol(BandMatrix<ScalarT, 11> * A, const ScalarT* b, ScalarT* x);

//typedef BandMatrix<Scalar, 7> HessianMatrix;
typedef BandMatrix<Scalar, 11> HessianMatrix;

#endif // !BAND_MATRIX_H
