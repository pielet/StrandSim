#ifndef BAND_MATRIX_H
#define BAND_MATRIX_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <vector>
#include "../Utils/Cublas.h"
#include "../Utils/CUDAMathDef.fwd.h"

const int BAND_WIDTH = 10;

template <typename ScalarT>
struct MatrixWrapper
{
	ScalarT* m_value;
	int* m_rowPtr;
	int* m_colIdx;
	MatrixWrapper(ScalarT* value, int* rowPtr, int* colIdx) :
		m_value(value), m_rowPtr(rowPtr), m_colIdx(colIdx) {}
	CUDA_CALLABLE_MEMBER void add(int x, int y, ScalarT value);
	CUDA_CALLABLE_MEMBER ScalarT get(int x, int y) const;
};

template <int w = BAND_WIDTH>
struct BandMatrixWrapper
{
	Scalar* values;
	int n_col;
	CUDA_CALLABLE_MEMBER BandMatrixWrapper(Scalar* a, int col) :values(a), n_col(col) {}
	CUDA_CALLABLE_MEMBER Scalar& operator[](int i) 
	{
		return values[i];
	}
	CUDA_CALLABLE_MEMBER Scalar operator[](int i) const
	{
		return values[i];
	}
	CUDA_CALLABLE_MEMBER int idx(int i, int j) const 
	{ 
		return (j - i + w) * n_col + i; 
	}
	CUDA_CALLABLE_MEMBER Scalar& operator()(int i, int j)
	{
		return values[idx(i, j)];
	}
	CUDA_CALLABLE_MEMBER Scalar operator()(int i, int j) const
	{
		return values[idx(i, j)];
	}
};

template <typename ScalarT, int w>
class BandMatrix
{
public:
	BandMatrix(int num_dofs, int num_strands, const std::vector<int>& strand_ptr, const int* dev_strand_ptr);
	~BandMatrix();

	ScalarT* getValue() { return m_value; }
	int* getRowPtr() { return m_rowPtr; }
	int* getColIdx() { return m_colIdx; }
	MatrixWrapper<ScalarT>* getMatrix() { return m_matrix; }

	ScalarT* getBandValues() { return m_bandValues; }

	int getn() const { return m_n; }
	int getnnz() const { return m_nnz; }

	void scale(ScalarT t);
	void addInDiagonal(ScalarT* m);
	void invDiagonal(ScalarT* diag) const;
	void setZero();

	void convertToBand();

protected:
	int m_ns;
	int m_n;	//< total dof
	int m_nnz;	//< non-zero
	int m_blocksPerGrid;

	cublasHandle_t m_cublasHandle;
	cublasHandle_t m_bandValueHandle;

	ScalarT* m_value;
	int* m_rowPtr;
	int* m_colIdx;
	MatrixWrapper<ScalarT>* m_matrix;

	ScalarT* m_bandValues;
	const int* m_strand_ptr;
};

//typedef BandMatrix<Scalar, 7> HessianMatrix;
typedef BandMatrix<Scalar, 11> HessianMatrix;

#endif // !BAND_MATRIX_H
