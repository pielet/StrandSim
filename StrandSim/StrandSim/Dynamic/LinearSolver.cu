#include "LinearSolver.h"
#include "../Utils/CUDAMathDef.h"
#include "BandMatrix.h"

#include <iostream>
#include <iomanip>
#include "../Utils/EigenDef.h"

const Scalar eps = 1e-12;

template <>
struct CusparseCaller<double>
{
	static void createCsr(cusparseSpMatDescr_t* mat, int n, int nnz, int* rowPtr, int* colIdx, double* values)
	{
		checkCudaErrors(cusparseCreateCsr(
			mat, n, n, nnz, rowPtr, colIdx, values,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	}

	static void createDnVec(cusparseDnVecDescr_t* vec, int n, double* values)
	{
		checkCudaErrors(cusparseCreateDnVec(vec, n, values, CUDA_R_64F));
	}

	static void mv_bufferSize(cusparseHandle_t handle, const double* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecAx,
		const double* beta, cusparseDnVecDescr_t vecx, size_t* bufferSize)
	{
		checkCudaErrors(cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecAx, beta, vecx,
			CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, bufferSize));
	}

	static void mv(cusparseHandle_t handle, const double* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecx,
		const double* beta, cusparseDnVecDescr_t vecAx, void* buffer)
	{
		checkCudaErrors(cusparseSpMV(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecx, beta, vecAx,
			CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, buffer));
	}

	static void ilu_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values, 
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, int* bufferSize)
	{
		checkCudaErrors(cusparseDcsrilu02_bufferSize(handle, n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
	}

	static void ilu_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const double* values, 
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseDcsrilu02_analysis(handle, n, nnz, descr, values, rowPtr, colIdx, info,
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void ilu(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values,
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseDcsrilu02(handle, n, nnz, descr, values, rowPtr, colIdx, info,
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void sv2_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, double* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, int* bufferSize)
	{
		checkCudaErrors(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
	}

	static void sv2_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const double* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
			descr, values, rowPtr, colIdx, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void sv2_solve(cusparseHandle_t handle, int n, int nnz, const double* alpha, const cusparseMatDescr_t descr, const double* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, const double* x, double* y, void* buffer)
	{
		checkCudaErrors(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, alpha,
			descr, values, rowPtr, colIdx, info, x, y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}
};

template <>
struct CusparseCaller<float>
{
	static void createCsr(cusparseSpMatDescr_t* mat, int n, int nnz, int* rowPtr, int* colIdx, float* values)
	{
		checkCudaErrors(cusparseCreateCsr(
			mat, n, n, nnz, rowPtr, colIdx, values,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	}

	static void createDnVec(cusparseDnVecDescr_t* vec, int n, float* values)
	{
		checkCudaErrors(cusparseCreateDnVec(vec, n, values, CUDA_R_32F));
	}

	static void mv_bufferSize(cusparseHandle_t handle, const float* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecAx,
		const float* beta, cusparseDnVecDescr_t vecx, size_t* bufferSize)
	{
		checkCudaErrors(cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecAx, &beta, vecx,
			CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, bufferSize));
	}

	static void mv(cusparseHandle_t handle, const float* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecx,
		const float* beta, cusparseDnVecDescr_t vecAx, void* buffer)
	{
		checkCudaErrors(cusparseSpMV(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA,
			vecx, beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, buffer));
	}

	static void ilu_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, int* bufferSize)
	{
		checkCudaErrors(cusparseScsrilu02_bufferSize(handle, n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
	}

	static void ilu_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const float* values,
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseScsrilu02_analysis(handle, n, nnz, descr, values, rowPtr, colIdx, info,
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void ilu(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
		const int* rowPtr, const int* colIdx, csrilu02Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseScsrilu02(handle, n, nnz, descr, values, rowPtr, colIdx, info,
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void sv2_bufferSize(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, float* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, int* bufferSize)
	{
		checkCudaErrors(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, nnz, descr, values, rowPtr, colIdx, info, bufferSize));
	}

	static void sv2_analysis(cusparseHandle_t handle, int n, int nnz, const cusparseMatDescr_t descr, const float* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, void* buffer)
	{
		checkCudaErrors(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
			descr, values, rowPtr, colIdx, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}

	static void sv2_solve(cusparseHandle_t handle, int n, int nnz, const float* alpha, const cusparseMatDescr_t descr, const float* values,
		const int* rowPtr, const int* colIdx, csrsv2Info_t info, const float* x, float* y, void* buffer)
	{
		checkCudaErrors(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, alpha,
			descr, values, rowPtr, colIdx, info, x, y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer));
	}
};

template <>
struct CusolverCaller<double>
{
	static void cholesky(cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t descrA, const double* values, const int* rowPtr, const int* colIdx, const double* b, double* x, int* singularity)
	{
		cusolverSpDcsrlsvchol(handle, n, nnz, descrA, values, rowPtr, colIdx, b, eps, 2, x, singularity);  // symamd
		//Eigen::VecXx test(nnz);
		//cudaMemcpy(test.data(), values, nnz * sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << test << "\n\n\n";
		//test.resize(n);
		//cudaMemcpy(test.data(), b, n * sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << test;
	}
};

template <>
struct CusolverCaller<float>
{
	static void cholesky(cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t descrA, const float* values, const int* rowPtr, const int* colIdx, const float* b, float* x, int* singularity)
	{
		cusolverSpScsrlsvchol(handle, n, nnz, descrA, values, rowPtr, colIdx, b, eps, 2, x, singularity);  // symamd
	}
};

// w = 10
// shared_mem size = 2 * (4 * max_vertex - 1)
__global__ void conjugateGradientKernel(int w, Scalar eps, const int* strand_ptr, const Scalar* A_in, const Scalar* b_in, Scalar* x_out)//, int* iters, Scalar* res, Scalar* res_detail)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;

	int n_dof = 4 * (strand_ptr[sid + 1] - strand_ptr[sid]) - 1;
	int start_dof = 4 * strand_ptr[sid] - sid;
	if (tid >= n_dof) return;

	extern __shared__ Scalar shared_mem[];
	Scalar* p = shared_mem + n_dof;
	Scalar* A = shared_mem + 2 * n_dof;

	Scalar xi, ri, Api, bi;
	Scalar rz, bnorm, alpha = 0., beta = 0.;
	int col_idx, i, j;

	bi = b_in[start_dof + tid];
	xi = x_out[start_dof + tid];
	int A_base = start_dof * (2 * w + 1);
	for (j = 0; j < 2 * w + 1; ++j) {
		A[j * n_dof + tid] = A_in[A_base + j * n_dof + tid];
	}
	Scalar inv_diag = 1. / A[w * n_dof + tid];

	// r0 = b - Ax
	p[tid] = xi;
	__syncthreads();
	Api = 0;
	for (j = 0; j < 2 * w + 1; ++j)
	{
		col_idx = tid - w + j;
		if (col_idx >= 0 && col_idx < n_dof) {
			Api += A[j * n_dof + tid] * p[col_idx];
			//Api += A_in[A_base + j * n_dof + tid] * p[col_idx];
		}
	}
	ri = bi - Api;

	// |b|
	shared_mem[tid] = bi * bi;
	__syncthreads();
	reduction(tid, n_dof, shared_mem);
	bnorm = shared_mem[0];
	__syncthreads();

	// xi, p, rz
	p[tid] = ri * inv_diag;
	shared_mem[tid] = p[tid] * ri;
	__syncthreads();
	reduction(tid, n_dof, shared_mem);
	rz = shared_mem[0];
	__syncthreads();

	for (i = 0; i < n_dof; ++i)
	{
		//res_detail[n_dof * n_dof * sid * 5 + i * n_dof * 5 + tid] = xi;
		//res_detail[n_dof * n_dof * sid * 5 + i * n_dof * 5 + n_dof + tid] = ri;
		//res_detail[n_dof * n_dof * sid * 5 + i * n_dof * 5 + n_dof * 2 + tid] = Api;
		//res_detail[n_dof * n_dof * sid * 5 + i * n_dof * 5 + n_dof * 3 + tid] = rz;
		//res_detail[n_dof * n_dof * sid * 5 + i * n_dof * 5 + n_dof * 4 + tid] = p[tid];

		// Ap
		Api = 0;
		for (j = 0; j < 2 * w + 1; ++j)
		{
			col_idx = tid - w + j;
			if (col_idx >= 0 && col_idx < n_dof) {
				Api += A[j * n_dof + tid] * p[col_idx];
				//Api += A_in[A_base + j * n_dof + tid] * p[col_idx];
			}
		}

		// alpha
		shared_mem[tid] = p[tid] * Api;
		__syncthreads();
		reduction(tid, n_dof, shared_mem);
		alpha = rz / shared_mem[0];
		__syncthreads();  // sync before changing contents, avoid broatcasting wrong number

		xi += alpha * p[tid];
		ri -= alpha * Api;

		// beta
		shared_mem[tid] = ri * ri * inv_diag;
		__syncthreads();
		reduction(tid, n_dof, shared_mem);

		beta = shared_mem[0] / rz;
		rz = shared_mem[0];
		if (rz < eps * bnorm) break;
		__syncthreads();

		//shared_mem[tid] = ri * ri;
		//__syncthreads();
		//reduction(tid, n_dof, shared_mem);
		//if (tid == 0) res[start_dof + i] = shared_mem[tid];
		//__syncthreads();

		//shared_mem[tid] = p[tid];
		//p[tid] = xi;
		//__syncthreads();
		//Api = 0;
		//for (j = 0; j < 2 * w + 1; ++j)
		//{
		//	col_idx = tid - w + j;
		//	if (col_idx >= 0 && col_idx < n_dof) {
		//		Api += A[j * n_dof + tid] * p[col_idx];
		//		//Api += A_in[A_base + j * n_dof + tid] * p[col_idx];
		//	} 
		//}
		//res_detail[n_dof * n_dof * sid + i * n_dof + tid] = bi - Api;
		//p[tid] = shared_mem[tid];
		//shared_mem[tid] = (bi - Api) * (bi - Api);
		//__syncthreads();
		//reduction(tid, n_dof, shared_mem);
		//if (tid == 0) res[start_dof + i] = shared_mem[tid];

		p[tid] = beta * p[tid] + ri * inv_diag;
		__syncthreads();
	}

	x_out[start_dof + tid] = xi;

	//if (tid == 0)
	//{
	//	iters[sid] = i;
	//	//res[sid] = shared_mem[tid];
	//}
}


LinearSolver::LinearSolver(int ns, const int* strand_ptr, HessianMatrix& matrix, PrecondT pt)
	: m_n(matrix.getn()), m_matrix(matrix), m_numStrands(ns), m_strand_ptr(strand_ptr)
{
	switch (pt)
	{
	case LinearSolver::NoPreconditionner:
		m_precond = new DummyPreconditioner(m_n);
		break;
	case LinearSolver::Diagnal:
		m_precond = new DiagnalPreconditioner(matrix);
		break;
	case LinearSolver::Factorization:
		m_precond = new FactorizationPreconditioner(matrix);
		break;
	}

	/* Create context */
	checkCudaErrors(cublasCreate(&m_cublasHandle));
	checkCudaErrors(cusparseCreate(&m_cusparseHandle));
	checkCudaErrors(cusolverSpCreate(&m_cusolverSpHandle));

	// LLT
	checkCudaErrors(cusparseCreateMatDescr(&m_descrA));
	checkCudaErrors(cusparseSetMatType(m_descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_descrA, CUSPARSE_INDEX_BASE_ZERO));

	/* Memory */
	checkCudaErrors(cudaMalloc((void**)& d_r, m_n * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& d_p, m_n * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& d_z, m_n * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& d_Ap, m_n * sizeof(Scalar)));

	// Creates cuSPARSE generic API objects
	CusparseCaller<Scalar>::createCsr(&m_matA, m_n, matrix.getnnz(), matrix.getRowPtr(), matrix.getColIdx(), matrix.getValue());
	CusparseCaller<Scalar>::createDnVec(&m_vecp, m_n, d_p);
	CusparseCaller<Scalar>::createDnVec(&m_vecAp, m_n, d_Ap);

	// Allocates workspace for cuSPARSE
	size_t bufferSize = 0;
	Scalar one = 1.0, zero = 0.0;
	CusparseCaller<Scalar>::mv_bufferSize(m_cusparseHandle, &one, m_matA, m_vecAp, &zero, m_vecp, &bufferSize);

	checkCudaErrors(cudaMalloc(&d_buffer, bufferSize));
}

LinearSolver::~LinearSolver()
{
	cublasDestroy(m_cublasHandle);
	cusparseDestroy(m_cusparseHandle);
	cusolverSpDestroy(m_cusolverSpHandle);
	cusparseDestroyMatDescr(m_descrA);

	if (m_matA) checkCudaErrors(cusparseDestroySpMat(m_matA));
	if (m_vecp) checkCudaErrors(cusparseDestroyDnVec(m_vecp));
	if (m_vecAp) checkCudaErrors(cusparseDestroyDnVec(m_vecAp));

	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_z);
	cudaFree(d_Ap);

	cudaFree(d_buffer);
}

bool LinearSolver::myCG(const Scalar* b, Scalar* x, int n_dof)
{
	//size_t shared_mem_size;
	//cudaOccupancyAvailableDynamicSMemPerBlock(&shared_mem_size, (void*)conjugateGradientKernel, m_numStrands, (g_maxVertex + g_threadsPerBlock - 1) / g_threadsPerBlock * g_threadsPerBlock);
	//std::cout << "Dynamic Shared Mempry size: " << shared_mem_size << std::endl;

	//int* d_iters;
	//Scalar* d_res;
	//Scalar* d_res_detail;
	//Eigen::VecXx band_values(m_n * 21);
	//cudaMalloc((void**)&d_iters, m_numStrands * sizeof(int));
	//cudaMalloc((void**)&d_res, m_numStrands * n_dof * sizeof(Scalar));
	//cudaMalloc((void**)&d_res_detail, 5 * m_numStrands * n_dof * n_dof * sizeof(Scalar));
	//cudaMemcpy(band_values.data(), m_matrix.getBandValues(), band_values.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "|dA|: ";
	//for (int i = 0; i < m_numStrands; ++i)
	//{
	//	std::cout << (band_values.segment(0, n_dof * 21) - band_values.segment(i * n_dof * 21, n_dof * 21)).squaredNorm() << ' ';
	//}
	//std::cout << std::endl;

	conjugateGradientKernel <<< m_numStrands, (4 * g_maxVertex - 1 + g_threadsPerBlock - 1) / g_threadsPerBlock * g_threadsPerBlock, (2 * BAND_WIDTH + 3) * (4 * g_maxVertex - 1) * sizeof(Scalar) >>> 
		(BAND_WIDTH, eps, m_strand_ptr, m_matrix.getBandValues(), b, x);// , d_iters, d_res, d_res_detail);
	//std::vector<Scalar> res(m_numStrands * n_dof);
	//cudaMemcpy(res.data(), d_res, res.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//Eigen::VecXx res_detail(m_numStrands * n_dof * n_dof * 5);
	//cudaMemcpy(res_detail.data(), d_res_detail, res_detail.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::vector<int> iters(m_numStrands);
	//cudaMemcpy(iters.data(), d_iters, iters.size() * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < m_numStrands; ++i)
	//{
	//	for (int j = 0; j < iters[i]; ++j)
	//	{
	//		std::cout << " strand " << i << " iter " << j << " res: "
	//			<< std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << res[n_dof * i + j] << "\n  ";
	//		//for (int k = 0; k < n_dof; ++k) {
	//		//	std::cout << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << res_detail[n_dof * n_dof * i + j * n_dof + k] << ' ';
	//		//}
	//		std::cout << '\n';
	//	}
	//}
	//if (std::isnan(res[m_numStrands * n_dof - 1])) {
	//	//for (int i = 0; i < m_numStrands; ++i)
	//	//{
	//	//	for (int j = 0; j < iters[i]; ++j)
	//	//	{
	//	//		std::cout << " strand " << i << " iter " << j << " res: "
	//	//			<< std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << res[n_dof * i + j] << "\n  ";
	//	//		//for (int k = 0; k < n_dof; ++k) {
	//	//		//	std::cout << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << res_detail[n_dof * n_dof * i + j * n_dof + k] << ' ';
	//	//		//}
	//	//		std::cout << '\n';
	//	//	}
	//	//}
	//	return false;
	//}
	//else return true;
	//Eigen::VecXx h_x(n_dof * m_numStrands);
	//cudaMemcpy(h_x.data(), x, h_x.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "|dx|: ";
	//Scalar delta_x;
	//bool flag = false;
	//int wrong_sid;
	//for (int i = 0; i < m_numStrands; ++i) {
	//	delta_x = (h_x.segment(n_dof * i, n_dof) - h_x.segment(0, n_dof)).squaredNorm();
	//	std::cout << delta_x << ' ';
	//	if (delta_x > 0) {
	//		flag = true;
	//		wrong_sid = i;
	//	}
	//}
	//std::cout << std::endl;

	//cudaMemcpy(h_x.data(), b, h_x.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "|db|: ";
	//for (int i = 0; i < m_numStrands; ++i) {
	//	delta_x = (h_x.segment(n_dof * i, n_dof) - h_x.segment(0, n_dof)).squaredNorm();
	//	std::cout << delta_x << ' ';
	//}
	//std::cout << std::endl;

	//if (flag)
	//{
	//	for (int j = 0; j < n_dof; ++j)
	//	{
	//		std::cout << "strand: 0 / " << wrong_sid << " res: " << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1)
	//			//<< res[j] << " / " << res[wrong_sid * n_dof + j] 
	//			<< " delta_x: " 
	//			<< (res_detail.segment(5 * wrong_sid * n_dof * n_dof + 5 * j * n_dof, n_dof) - res_detail.segment(5 * j * n_dof, n_dof)).squaredNorm()
	//			<< " delta_r: "
	//			<< (res_detail.segment(5 * wrong_sid * n_dof * n_dof + 5 * j * n_dof + n_dof, n_dof) - res_detail.segment(5 * j * n_dof + n_dof, n_dof)).squaredNorm()
	//			<< " Api: "
	//			<< (res_detail.segment(5 * wrong_sid * n_dof * n_dof + 5 * j * n_dof + 2 * n_dof, n_dof) - res_detail.segment(5 * j * n_dof + 2 * n_dof, n_dof)).squaredNorm()
	//			<< " rz: "
	//			<< (res_detail.segment(5 * wrong_sid * n_dof * n_dof + 5 * j * n_dof + 3 * n_dof, n_dof) - res_detail.segment(5 * j * n_dof + 3 * n_dof, n_dof)).squaredNorm()
	//			<< " delta_p: "
	//			<< (res_detail.segment(5 * wrong_sid * n_dof * n_dof + 5 * j * n_dof + 4 * n_dof, n_dof) - res_detail.segment(5 * j * n_dof + 4 * n_dof, n_dof)).squaredNorm()
	//			<< std::endl;

	//		//for (int k = 0; k < n_dof; ++k) {
	//		//	std::cout << res_detail[j * n_dof + k] << ' ';
	//		//}
	//		//std::cout << '\n';
	//		//for (int k = 0; k < n_dof; ++k) {
	//		//	std::cout << res_detail[wrong_sid * n_dof * n_dof + j * n_dof + k] << ' ';
	//		//}
	//		//std::cout << '\n';
	//	}
	//	exit(-1);
	//}

/*	cudaFree(d_iters);
	cudaFree(d_res);
	cudaFree(d_res_detail)*/
	return true;
}


bool LinearSolver::cholesky(const Scalar* b, Scalar* x)
{
	int singularity;
	CusolverCaller<Scalar>::cholesky(m_cusolverSpHandle, m_n, m_matrix.getnnz(), m_descrA, m_matrix.getValue(),
		m_matrix.getRowPtr(), m_matrix.getColIdx(), b, x, &singularity);
	return singularity < 0;
}


bool LinearSolver::conjugateGradient(const Scalar* d_b, Scalar* x)
{
	Scalar one = 1.0, zero = 0.0, neg_one = -1.0;
	Scalar res, bnorm, alpha, beta;
	Scalar pAp, rz, old_rz;

	int nnz = m_matrix.getnnz();
	Scalar* values = m_matrix.getValue();
	const int* rowPtr = m_matrix.getRowPtr();
	const int* colIdx = m_matrix.getColIdx();

	// Perform analysis for ILU
	bool status = m_precond->analysis();
	if (!status)
	{
		std::cerr << "Preconditioner analysis failed. EXIT." << std::endl;
		exit(-1);
	}

	// r0 = b - Ax
	CublasCaller<Scalar>::copy(m_cublasHandle, m_n, x, d_p);
	CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_b, d_r);

	CusparseCaller<Scalar>::mv(m_cusparseHandle, &one, m_matA, m_vecp, &zero, m_vecAp, &d_buffer);
	CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &neg_one, d_Ap, d_r);

	CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
	CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_b, d_b, &bnorm);

	//if (res / bnorm < eps) return;

	m_precond->solve(d_r, d_z);
	CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_z, d_p);
	CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_z, &rz);

	Eigen::VecXx test(m_n);

	int k = 0;
	for (k; k < m_n/m_numStrands; ++k)
	{
		CusparseCaller<Scalar>::mv(m_cusparseHandle, &one, m_matA, m_vecp, &zero, m_vecAp, d_buffer);
		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_p, d_Ap, &pAp);
		alpha = rz / pAp;

		CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &alpha, d_p, x);
		alpha = -alpha;
		CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &alpha, d_Ap, d_r);
		old_rz = rz;

		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
		//std::cout << "\t iter: " << k << " rTr: " << res << std::endl;

		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_p, d_Ap);
		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, x, d_p);
		//CusparseCaller<Scalar>::mv(m_cusparseHandle, &one, m_matA, m_vecp, &zero, m_vecAp, d_buffer);
		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_r, d_z);
		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_b, d_r);
		//CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &neg_one, d_Ap, d_r);
		//CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_r, &res);
		//cudaMemcpy(test.data(), x, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < m_n; ++i)
		//	std::cout << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1) << test(i) << ' ';
		//std::cout << '\n';
		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_Ap, d_p);
		//CublasCaller<Scalar>::copy(m_cublasHandle, m_n, d_z, d_r);

		//if (res < eps * bnorm) break;

		m_precond->solve(d_r, d_z);
		CublasCaller<Scalar>::dot(m_cublasHandle, m_n, d_r, d_z, &rz);
		beta = rz / old_rz;
		CublasCaller<Scalar>::scal(m_cublasHandle, m_n, &beta, d_p);
		CublasCaller<Scalar>::axpy(m_cublasHandle, m_n, &one, d_z, d_p);
	}

	std::cout << "Total CG iteration: " << k << " residual: " << res << std::endl;

	return true;
}

void DummyPreconditioner::solve(const Scalar* in, Scalar* out)
{
	cudaMemcpy(out, in, m_n * sizeof(Scalar), cudaMemcpyDeviceToDevice);
}

DiagnalPreconditioner::DiagnalPreconditioner(HessianMatrix& A) :m_A(A)
{
	checkCudaErrors(cudaMalloc((void**)& m_invDiagA, A.getn() * sizeof(Scalar)));
}

DiagnalPreconditioner::~DiagnalPreconditioner()
{
	cudaFree(m_invDiagA);
}

bool DiagnalPreconditioner::analysis()
{
	m_A.invDiagonal(m_invDiagA);
	return true;
}

void DiagnalPreconditioner::solve(const Scalar* in, Scalar* out)
{
	int block = (m_A.getn() + g_threadsPerBlock - 1) / g_threadsPerBlock;
	cwiseMultiply <<< block, g_threadsPerBlock >>> (m_A.getn(), m_invDiagA, in, out);
}

FactorizationPreconditioner::FactorizationPreconditioner(HessianMatrix& A) :m_A(A)
{
	checkCudaErrors(cusparseCreate(&m_cusparseHandle));

	checkCudaErrors(cusparseCreateMatDescr(&m_descrA));
	checkCudaErrors(cusparseSetMatType(m_descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_descrA, CUSPARSE_INDEX_BASE_ZERO));

	// Creates ILU info and triangular solve info
	checkCudaErrors(cusparseCreateCsrilu02Info(&m_infoILU));

	checkCudaErrors(cusparseCreateCsrsv2Info(&m_infoL));
	checkCudaErrors(cusparseCreateCsrsv2Info(&m_infoU));

	checkCudaErrors(cusparseCreateMatDescr(&m_descrL));
	checkCudaErrors(cusparseSetMatType(m_descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_descrL, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER));
	checkCudaErrors(cusparseSetMatDiagType(m_descrL, CUSPARSE_DIAG_TYPE_UNIT));

	checkCudaErrors(cusparseCreateMatDescr(&m_descrU));
	checkCudaErrors(cusparseSetMatType(m_descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_descrU, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(cusparseSetMatFillMode(m_descrU, CUSPARSE_FILL_MODE_UPPER));
	checkCudaErrors(cusparseSetMatDiagType(m_descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

	checkCudaErrors(cudaMalloc((void**)& m_valsILU, A.getnnz() * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_y, A.getn() * sizeof(Scalar)));

	size_t bufferSize = 0;
	int tmp;

	CusparseCaller<Scalar>::ilu_bufferSize(m_cusparseHandle, A.getn(), A.getnnz(), m_descrA,
		A.getValue(), A.getRowPtr(), A.getColIdx(), m_infoILU, &tmp);
	if (tmp > bufferSize) bufferSize = tmp;

	CusparseCaller<Scalar>::sv2_bufferSize(m_cusparseHandle, A.getn(), A.getnnz(), m_descrL,
		A.getValue(), A.getRowPtr(), A.getColIdx(), m_infoL, &tmp);
	if (tmp > bufferSize) bufferSize = tmp;

	CusparseCaller<Scalar>::sv2_bufferSize(m_cusparseHandle, A.getn(), A.getnnz(), m_descrU,
		A.getValue(), A.getRowPtr(), A.getColIdx(), m_infoU, &tmp);
	if (tmp > bufferSize) bufferSize = tmp;

	checkCudaErrors(cudaMalloc(&m_buffer, bufferSize));
}

FactorizationPreconditioner::~FactorizationPreconditioner()
{
	cusparseDestroy(m_cusparseHandle);
	cusparseDestroyMatDescr(m_descrA);

	cusparseDestroyCsrilu02Info(m_infoILU);
	cusparseDestroyCsrsv2Info(m_infoL);
	cusparseDestroyCsrsv2Info(m_infoU);
	cusparseDestroyMatDescr(m_descrL);
	cusparseDestroyMatDescr(m_descrU);

	cudaFree(m_valsILU);
	cudaFree(m_y);
	cudaFree(m_buffer);
}

bool FactorizationPreconditioner::analysis()
{
	int n = m_A.getn(), nnz = m_A.getnnz();
	Scalar* values = m_A.getValue();
	const int* rowPtr = m_A.getRowPtr();
	const int* colIdx = m_A.getColIdx();

	int structural_zero, numerical_zero;

	// Perform analysis for ILU
	CusparseCaller<Scalar>::ilu_analysis(m_cusparseHandle, n, nnz, m_descrA, values, rowPtr, colIdx, m_infoILU, m_buffer);

	auto status = cusparseXcsrilu02_zeroPivot(m_cusparseHandle, m_infoILU, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	}

	cudaMemcpy(m_valsILU, values, nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice);
	CusparseCaller<Scalar>::ilu(m_cusparseHandle, n, nnz, m_descrA, m_valsILU, rowPtr, colIdx, m_infoILU, m_buffer);

	status = cusparseXcsrilu02_zeroPivot(m_cusparseHandle, m_infoILU, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
	}

	// Perform analysis for trianguler
	CusparseCaller<Scalar>::sv2_analysis(m_cusparseHandle, n, nnz, m_descrL, m_valsILU, rowPtr, colIdx, m_infoL, m_buffer);
	CusparseCaller<Scalar>::sv2_analysis(m_cusparseHandle, n, nnz, m_descrU, m_valsILU, rowPtr, colIdx, m_infoU, m_buffer);

	return structural_zero < 0 && numerical_zero < 0;
}

void FactorizationPreconditioner::solve(const Scalar* in, Scalar* out)
{
	Scalar one = 1.0;

	// out = U^-1 * L^-1 * in
	CusparseCaller<Scalar>::sv2_solve(m_cusparseHandle, m_A.getn(), m_A.getnnz(), &one, m_descrL, 
		m_valsILU, m_A.getRowPtr(), m_A.getColIdx(), m_infoL, in, m_y, m_buffer);
	CusparseCaller<Scalar>::sv2_solve(m_cusparseHandle, m_A.getn(), m_A.getnnz(), &one, m_descrU, 
		m_valsILU, m_A.getRowPtr(), m_A.getColIdx(), m_infoU, m_y, out, m_buffer);
}