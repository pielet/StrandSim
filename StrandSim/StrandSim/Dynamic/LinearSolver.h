#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include <cusolverSp.h>
#include <cusparse.h>
#include "../Utils/Cublas.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "BandMatrix.h"

class PreconditionerBase;

class LinearSolver
{
public:
	enum PrecondT { NoPreconditionner, Diagnal, Factorization };

	LinearSolver(int ns, const int* strand_ptr, HessianMatrix& matrix, PrecondT pt = NoPreconditionner);
	~LinearSolver();

	bool cholesky(const Scalar* b, Scalar* x);
	bool conjugateGradient(const Scalar* b, Scalar* x);
	bool myCG(const Scalar* b, Scalar* x, int n_dof);
	bool stupidConjugateGradient(int n_strand, const int* strand_ptr, const Scalar* b, Scalar* x);

private:
	int m_n;
	int m_numStrands;
	const int* m_strand_ptr;

	HessianMatrix& m_matrix;
	PreconditionerBase* m_precond;

	cublasHandle_t m_cublasHandle;
	cusparseHandle_t m_cusparseHandle;
	cusolverSpHandle_t m_cusolverSpHandle;

	// CG
	cusparseMatDescr_t m_descrA;
	cusparseSpMatDescr_t m_matA;
	cusparseDnVecDescr_t m_vecp;
	cusparseDnVecDescr_t m_vecAp;

	/* Device pointer */
	Scalar* d_r;
	Scalar* d_p;
	Scalar* d_z;
	Scalar* d_Ap;

	void* d_buffer;
};

// Preconditioner
class PreconditionerBase
{
public:
	virtual ~PreconditionerBase() {};
	virtual bool analysis() = 0;
	virtual void solve(const Scalar* in, Scalar* out) = 0;
};

class DummyPreconditioner :public PreconditionerBase
{
public:
	DummyPreconditioner(int n) :m_n(n) {}
	virtual bool analysis() { return true; }
	virtual void solve(const Scalar* in, Scalar* out);
private:
	int m_n;
};

class DiagnalPreconditioner :public PreconditionerBase
{
public:
	DiagnalPreconditioner(HessianMatrix& A);
	~DiagnalPreconditioner();
	virtual bool analysis();
	virtual void solve(const Scalar* in, Scalar* out);
private:
	HessianMatrix& m_A;
	Scalar* m_invDiagA;
};

class FactorizationPreconditioner :public PreconditionerBase
{
public:
	FactorizationPreconditioner(HessianMatrix& A);
	~FactorizationPreconditioner();
	virtual bool analysis();
	virtual void solve(const Scalar* in, Scalar* out);
private:
	HessianMatrix& m_A;

	Scalar* m_valsILU;
	Scalar* m_y;
	void* m_buffer;

	cusparseHandle_t m_cusparseHandle;

	cusparseMatDescr_t m_descrA;
	
	csrilu02Info_t m_infoILU;
	csrsv2Info_t m_infoL;
	csrsv2Info_t m_infoU;
	cusparseMatDescr_t m_descrL;
	cusparseMatDescr_t m_descrU;
};

template <typename ScalarT> struct CusparseCaller;
template <typename ScalarT> struct CusolverCaller;

#endif // !LINEAR_SOLVER_H
