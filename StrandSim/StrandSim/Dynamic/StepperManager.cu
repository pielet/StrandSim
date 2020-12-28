#include "StepperManager.h"
#include <algorithm>
#include <iomanip>
#include "../Utils/CUDAMathDef.h"

#include "../Utils/EigenDef.h"

__global__ void computeRhs(int n_dof, Scalar dt, const Scalar* mass, const Scalar* next_v, const Scalar* v, const Scalar* g, Scalar* rhs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n_dof) return;

	// rhs = M * (v_t + 1 - v_t) - h * (f_ext - \grad E)
	rhs[i] = mass[i] * (next_v[i] - v[i]) - dt * (g[i] + rhs[i]);
}

__global__ void add(int n_dof, Scalar* c, Scalar alpha, const Scalar* a, const Scalar* b)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n_dof) return;

	c[i] = alpha * a[i] + b[i];
}

__global__ void computeInertiaTerm(int n_dof, Scalar* inertia, const Scalar* mass, const Scalar* next_v, const Scalar* v, const Scalar* gravity, Scalar dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n_dof) return;

	Scalar delta_v = next_v[i] - (v[i] + dt * gravity[i] / mass[i]);
	inertia[i] = mass[i] * delta_v * delta_v;
}

StepperManager::StepperManager(Scalar dt, int n_dof, int n_fixed, const Scalar* x, const Scalar* v, const std::vector<int>& strand_ptr, 
	const int* fixed_idx, const Scalar* fixed_pos,const SimulationParameters* sim_params, 
	const StrandParameters* strand_params, const CollisionParameters* col_params) :
	m_time(0.),
	m_dt(dt),
	m_numDofs(n_dof),
	m_numStrands(strand_ptr.size() - 1),
	m_numVertices(strand_ptr.back()),
	m_numEdges(m_numVertices - m_numStrands),
	m_numInnerVtx(m_numVertices - 2 * m_numStrands),
	m_numFixed(n_fixed),
	m_simParams(sim_params),
	m_strandParams(strand_params),
	m_colParams(col_params),
	m_dofBlocks((n_dof + g_threadsPerBlock - 1) / g_threadsPerBlock),
	m_lsStep(1.0)
{
	checkCudaErrors(cublasCreate(&m_cublasHandle));

	// Allocates memory
	checkCudaErrors(cudaMalloc((void**)& m_strand_ptr, strand_ptr.size() * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_vtx_2_strand, m_numVertices * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_edge_2_strand, m_numEdges * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_inner_vtx_2_strand, m_numInnerVtx * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)& m_fixed_idx, m_numFixed * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)& m_x, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_v, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_savedV, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_referenceFrames1, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_tangents, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_fixedTargets, m_numFixed * sizeof(Vec3x)));

	checkCudaErrors(cudaMalloc((void**)& m_tmp, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_descentDir, m_numDofs * sizeof(Scalar)));

	// Computes indices
	std::vector<int> vtx_2_strand(m_numVertices);
	std::vector<int> edge_2_strand(m_numEdges);
	std::vector<int> inner_vtx_2_strand(m_numInnerVtx);

	int j;
	for (int i = 0; i < m_numStrands; ++i)
	{
		for (j = strand_ptr[i]; j < strand_ptr[i + 1]; ++j)
		{
			vtx_2_strand[j] = i;
		}
		for (j = strand_ptr[i]; j < strand_ptr[i + 1] - 1; ++j)
		{
			edge_2_strand[j - i] = i;
		}
		for (j = strand_ptr[i]; j < strand_ptr[i + 1] - 2; ++j)
		{
			inner_vtx_2_strand[j - 2 * i] = i;
		}
	}

	m_maxVtxPerStrand = 0;
	for (j = 0; j < m_numStrands; ++j)
	{
		m_maxVtxPerStrand = std::max(m_maxVtxPerStrand, strand_ptr[j + 1] - strand_ptr[j]);
	}

	// Copy data
	cudaMemcpy(m_x, x, m_numDofs * sizeof(Scalar), cudaMemcpyHostToDevice);
	cudaMemcpy(m_v, v, m_numDofs * sizeof(Scalar), cudaMemcpyHostToDevice);

	cudaMemcpy(m_fixedTargets, fixed_pos, m_numFixed * sizeof(Vec3x), cudaMemcpyHostToDevice);
	cudaMemcpy(m_fixed_idx, fixed_idx, m_numFixed * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(m_strand_ptr, strand_ptr.data(), (m_numStrands + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_vtx_2_strand, vtx_2_strand.data(), m_numVertices * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_edge_2_strand, edge_2_strand.data(), m_numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_inner_vtx_2_strand, inner_vtx_2_strand.data(), m_numInnerVtx * sizeof(int), cudaMemcpyHostToDevice);
	
	m_A = new HessianMatrix(m_numDofs, m_numStrands, strand_ptr, m_strand_ptr);
	m_linearSolver = new LinearSolver(m_numStrands, m_strand_ptr, *m_A, LinearSolver::Diagnal);
	m_dynamics = new StrandStates(
		m_strandParams, m_numDofs, m_numStrands, m_numVertices, m_maxVtxPerStrand, m_numFixed, m_fixed_idx, m_strand_ptr, 
		m_vtx_2_strand, m_edge_2_strand, m_inner_vtx_2_strand, m_fixedTargets, m_A);

	//! TODO: memcpy and init computation can be parallel by streams
	cudaMemcpy(m_dynamics->getX(), x, m_numDofs * sizeof(Scalar), cudaMemcpyHostToDevice);
	m_dynamics->init();
	m_mass = m_dynamics->getMass();
	m_gravity = m_dynamics->getGravity();

	cudaMemcpy(m_referenceFrames1, m_dynamics->getReferenceFrames1(), m_numEdges * sizeof(Vec3x), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_tangents, m_dynamics->getTangents(), m_numEdges * sizeof(Vec3x), cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize();
}

StepperManager::~StepperManager()
{
	cublasDestroy(m_cublasHandle);

	cudaFree(m_fixed_idx);
	cudaFree(m_strand_ptr);
	cudaFree(m_vtx_2_strand);
	cudaFree(m_edge_2_strand);
	cudaFree(m_inner_vtx_2_strand);

	cudaFree(m_x);
	cudaFree(m_v);
	cudaFree(m_savedV);
	cudaFree(m_referenceFrames1);
	cudaFree(m_tangents);
	cudaFree(m_fixedTargets);

	cudaFree(m_tmp);
	cudaFree(m_descentDir);

	delete m_A;
	delete m_linearSolver;
	delete m_dynamics;
}

void StepperManager::step()
{
	m_timing.reset();

	std::cout << "Time: " << m_time << std::endl;
	m_time += m_dt;

	std::cout << "[Prepare Simulation Step]" << std::endl;
	m_timer.start();
	prepareStep();
	m_timing.prepare = m_timer.elapsedMilliseconds();

	std::cout << "[Step Dynamics]" << std::endl;
	m_timer.start();
	int k = 0;
	for (; k < m_simParams->m_maxNewtonIterations; ++k)
	{
		if (performNewtonStep()) break;
	}
	m_timing.dynamics = m_timing.dynamicsSum();

	std::cout << "[Post Dynamics]" << std::endl;
	m_timer.start();
	postStep();
	m_timing.post = m_timer.elapsedMilliseconds();

	m_timings.push_back(m_timing);
	printTiming();
}

void StepperManager::prepareStep()
{
	CublasCaller<Scalar>::copy(m_cublasHandle, m_numDofs, m_v, m_savedV);
	m_lsStep = 1.0;
}

bool StepperManager::performNewtonStep()
{
	m_timer.start();
	add << < m_dofBlocks, g_threadsPerBlock >> > (m_numDofs, m_dynamics->getX(), m_dt, m_v, m_x);

	//Eigen::VecXx test(m_numDofs);
	//Eigen::VecXx mats((m_numVertices - 2 * m_numStrands) * 16 * 9);
	//Eigen::VecXx mats1((m_numVertices - 2 * m_numStrands) * 16 * 9);
	//Eigen::VecXx vec((m_numVertices - 2 * m_numStrands) * 16 * 3);
	//Eigen::VecXx vec1((m_numVertices - 2 * m_numStrands) * 16 * 3);
	//Eigen::VecXx grad((m_numVertices - 2 * m_numStrands) * 7 * 11);
	//Eigen::VecXx grad1((m_numVertices - 2 * m_numStrands) * 7 * 11);
	//Eigen::VecXx localJ((m_numVertices - 2 * m_numStrands) * 121);
	//Eigen::VecXx localJ1((m_numVertices - 2 * m_numStrands) * 121);

	//Eigen::VecXx force(m_numDofs);
	//Eigen::VecXx force1(m_numDofs);

	m_dynamics->updateStates(m_tangents, m_referenceFrames1);
	m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(false, mats1, vec1, grad1, localJ1);
	m_dynamics->computeForcesAndJacobian(true);

	//Eigen::VecXx tmp_band(m_numDofs * 21);
	//cudaMemcpy(tmp_band.data(), m_A->getBandValues(), tmp_band.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//m_A->convertToBand();
	//test.resize(m_numDofs * 21);
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//int n_dof = m_numDofs / m_numStrands;
	////if ((tmp_band - test).squaredNorm() > 0) {
	////	m_A->setZero();
	//	m_dynamics->computeForcesAndJacobian(false, mats1, vec1, grad1, localJ1, false, false, true);
	//	cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(true, mats, vec, grad, localJ, false, false, true);
	////m_A->convertToBand();
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//std::cout << "A bend:\n";
	//std::cout.precision(15);
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//	}
	//	std::cout << '\n';
	//}
	//std::cout << "totalForces bend:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << force[i] << ' ';
	//std::cout << "\n\n";

	//m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(true, mats, vec, grad, localJ, true, false, false);
	////m_A->convertToBand();
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//std::cout << "A stretch:\n";
	//std::cout.precision(15);
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//	}
	//	std::cout << '\n';
	//}
	//std::cout << "totalForces stretch:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << force[i] << ' ';
	//std::cout << "\n\n";

	//m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(true, mats, vec, grad, localJ, false, true, false);
	////m_A->convertToBand();
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//std::cout << "A twist:\n";
	//std::cout.precision(15);
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//	}
	//	std::cout << '\n';
	//}
	//std::cout << "totalForces twist:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << force[i] << ' ';
	//std::cout << "\n\n";
	//	//for (int i = 0; i < m_numDofs; ++i)
	//	//	std::cout << force1[i] << ' ';
	//	//std::cout << std::endl;
	//	//force -= force1;
	//	//for (int i = 0; i < m_numDofs; ++i)
	//	//	std::cout << force[i] << ' ';
	//	//std::cout << std::endl;

	//		//for (int k = 0; k < m_numVertices - 2 * m_numStrands; ++k) {
	//		//	std::cout << "inner node: " << k << std::endl;
	//		//	for (int i = 0; i < 7; ++i) {
	//		//		for (int j = 0; j < 11; ++j)
	//		//			std::cout << grad[77 * k + 11 * i + j] << ' ';
	//		//		std::cout << '\n';
	//		//	}
	//		//	std::cout << '\n';
	//		//	for (int i = 0; i < 7; ++i) {
	//		//		for (int j = 0; j < 11; ++j)
	//		//			std::cout << grad1[77 * k + 11 * i + j] << ' ';
	//		//		std::cout << '\n';
	//		//	}
	//		//	std::cout << '\n';
	//		//	for (int i = 0; i < 7; ++i) {
	//		//		for (int j = 0; j < 11; ++j)
	//		//			std::cout << (grad - grad1)[77 * k + 11 * i + j] << ' ';
	//		//		std::cout << '\n';
	//		//	}
	//		//	std::cout << '\n';
	//		//}

	//	for (int k = 0; k < m_numVertices - 2 * m_numStrands; ++k) {
	//		std::cout << "inner node: " << k << std::endl;
	//		for (int i = 0; i < 16; ++i){
	//			for (int j = 0; j < 9; ++j)
	//				std::cout << mats1[16 * 9 * k + i * 9 + j] << ' ';
	//			std::cout << '\n';
	//			//for (int j = 0; j < 9; ++j)
	//			//	std::cout << mats[16 * 9 * k + i * 9 + j] << ' ';
	//			//std::cout << '\n';
	//			for (int j = 0; j < 9; ++j) {
	//				if (std::fabs(mats1[16 * 9 * k + i * 9 + j]) > 1e-40)
	//					std::cout << (mats - mats1)[16 * 9 * k + i * 9 + j] / mats1[16 * 9 * k + i * 9 + j] << ' ';
	//				else std::cout << "0 ";
	//			}
	//			std::cout << "\n\n";
	//			//printf("(%.15f, %.15f, %.15f)\n", vec[16 * 3 * k + 3 * i], vec[16 * 3 * k + 3 * i + 1], vec[16 * 3 * k + 3 * i + 2]);
	//		}
	//		std::cout << "local J: " << std::endl;
	//		for (int i = 0; i < 11; ++i) {
	//			for (int j = 0; j < 11; ++j)
	//				std::cout << localJ1[121 * k + 11 * i + j] << ' ';
	//			std::cout << '\n';
	//		}
	//		std::cout << '\n';
	//		//for (int i = 0; i < 11; ++i) {
	//		//	for (int j = 0; j < 11; ++j)
	//		//		std::cout << localJ[121 * k + 11 * i + j] << ' ';
	//		//	std::cout << '\n';
	//		//}
	//		//std::cout << '\n';
	//		for (int i = 0; i < 11; ++i) {
	//			for (int j = 0; j < 11; ++j) {
	//				if (std::fabs(localJ1[121 * k + 11 * i + j]) > 1e-40)
	//					std::cout << (localJ - localJ1)[121 * k + 11 * i + j] / localJ1[121 * k + 11 * i + j] << ' ';
	//				else std::cout << "0 ";
	//			}
	//			std::cout << '\n';
	//		}
	//		printf("\n\n");
	//	}
	////}
	//m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(false, mats, vec, grad, localJ);

	//m_A->convertToBand();
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//std::cout << "A total old:\n";
	//std::cout.precision(15);
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//	}
	//	std::cout << '\n';
	//}
	//std::cout << "totalForces old:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << force[i] << ' ';
	//std::cout << "\n\n";

	//m_A->setZero();
	//m_dynamics->computeForcesAndJacobian(true, mats, vec, grad, localJ);

	////m_A->convertToBand();
	//cudaMemcpy(test.data(), m_A->getBandValues(), test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(force.data(), m_dynamics->getTotalForces(), force.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//std::cout << "A total new:\n";
	//std::cout.precision(15);
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//	}
	//	std::cout << '\n';
	//}
	//std::cout << "totalForces new:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << force[i] << ' ';
	//std::cout << "\n\n";

	// Computes RHS
	Scalar* b = m_dynamics->getTotalForces();
	//cudaMemcpy(test.data(), b, test.size() * sizeof(double), cudaMemcpyDeviceToHost);
	//std::cout << test << "\n\n\n";
	computeRhs << < m_dofBlocks, g_threadsPerBlock >> > (m_numDofs, m_dt, m_mass, m_v, m_savedV, m_gravity, b);

	// Computes LHS
	m_A->scale(m_dt * m_dt);
	m_A->addInDiagonal(m_mass);
	m_timing.construct += m_timer.elapsedMilliseconds();
	m_timing.derivative += m_dynamics->getTiming();

	//Eigen::VecXx tmp_band(m_numDofs * 21);
	//cudaMemcpy(tmp_band.data(), m_A->getBandValues(), tmp_band.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);

	//int n_dof = m_numDofs / m_numStrands;
	//if ((test - tmp_band).squaredNorm() > 0) {
	//	for (int k = 0; k < m_numStrands; ++k) {
	//		for (int i = 0; i < 21; ++i)
	//		{
	//			for (int j = 0; j < n_dof; ++j) {
	//				if (square(test[21 * n_dof * k + i * n_dof + j] - tmp_band[21 * n_dof * k + i * n_dof + j]) > 1e-12)
	//					printf("(%d, %d, %d): %.10f / %.10f\n", k, i, j, test[21 * n_dof * k + i * n_dof + j], tmp_band[21 * n_dof * k + i * n_dof + j]);
	//			}
	//		}
	//	}
	//}
	//int n_dof = m_numDofs / m_numStrands;
	//for (int k = 0; k < m_numStrands; ++k) {
	//	for (int i = 0; i < 21; ++i) {
	//		for (int j = 0; j < n_dof; ++j)
	//			std::cout << test(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		std::cout << '\n';
	//		//for (int j = 0; j < n_dof; ++j)
	//		//	std::cout << tmp_band(k * 21 * n_dof + i * n_dof + j) << ' ';
	//		//std::cout << "\n\n";
	//	}
	//	std::cout << '\n';
	//}

	// Solver linear equation
	m_timer.start();
	Scalar neg_one = -1.0;
	cudaMemset(m_descentDir, 0, m_numDofs * sizeof(Scalar));
	//CublasCaller<Scalar>::scal(m_cublasHandle, m_numDofs, &neg_one, m_descentDir);  // initial value
	bool SPD = m_linearSolver->myCG(b, m_descentDir, m_numDofs / m_numStrands);
	//bool SPD = m_linearSolver->conjugateGradient(b, m_descentDir);
	//test.resize(m_numDofs);
	//cudaMemcpy(test.data(), m_descentDir, test.size() * sizeof(double), cudaMemcpyDeviceToHost);
	//std::cout << "descentDir:\n";
	//for (int i = 0; i < m_numDofs; ++i)
	//	std::cout << test[i] << ' ';
	//std::cout << std::endl;
	//bool SPD = m_linearSolver->cholesky(b, m_descentDir);
	if (!SPD)
	{
		std::cerr << "NOT SPD. EXIT.";
		exit(-1);
	}
	CublasCaller<Scalar>::scal(m_cublasHandle, m_numDofs, &neg_one, m_descentDir);
	m_timing.solveLinear += m_timer.elapsedMilliseconds();

	// Line-search
	m_timer.start();
	Scalar obj_value = lineSearch(m_v, b, m_descentDir);
	CublasCaller<Scalar>::axpy(m_cublasHandle, m_numDofs, &m_lsStep, m_descentDir, m_v);
	m_timing.lineSearch += m_timer.elapsedMilliseconds();

	// Check convergency
	m_timer.start();
	Scalar delta_v;
	CublasCaller<Scalar>::dot(m_cublasHandle, m_numDofs, m_descentDir, m_descentDir, &delta_v);
	delta_v = m_lsStep * sqrt(delta_v);
	m_timing.check += m_timer.elapsedMilliseconds();

	std::cout << "alpha: " << m_lsStep << "  obj value: " << obj_value << "  avg. delta v: " << delta_v / m_numDofs << std::endl;

	if (delta_v / m_numDofs < m_simParams->m_newtonTolerance)
		return true;
	else return false;
}

Scalar StepperManager::lineSearch(const Scalar* current_v, const Scalar* gradient_dir, const Scalar* descent_dir)
{
	Scalar current_obj_value = evaluateObjectValue(current_v);

	if (m_simParams->m_useLineSearch)
	{
		Scalar next_obj_value, rhs;
		Scalar grad_dot_desc;
		CublasCaller<Scalar>::dot(m_cublasHandle, m_numDofs, gradient_dir, descent_dir, &grad_dot_desc);

		m_lsStep = std::min(1., 2 * std::max(1e-5, m_lsStep)) / m_simParams->m_lsBeta;

		do {
			m_lsStep *= m_simParams->m_lsBeta;
			if (m_lsStep < 1e-5) break;

			add <<< m_dofBlocks, g_threadsPerBlock >>> (m_numDofs, m_tmp, m_lsStep, m_descentDir, current_v);
			next_obj_value = evaluateObjectValue(m_tmp);
			rhs = current_obj_value + m_simParams->m_lsAlpha * m_lsStep * grad_dot_desc;
		} while (next_obj_value > rhs);

		if (m_lsStep < 1e-5)
		{
			m_lsStep = 0;
			return current_obj_value;
		}
		else return next_obj_value;
	}
	else return 0;
}

Scalar StepperManager::evaluateObjectValue(const Scalar* v)
{
	add <<< m_dofBlocks, g_threadsPerBlock >>> (m_numDofs, m_dynamics->getX(), m_dt, v, m_x);
	m_dynamics->updateStates(m_tangents, m_referenceFrames1);
	Scalar elasticity = m_dynamics->computeEnergy();

	computeInertiaTerm <<< m_dofBlocks, g_threadsPerBlock >>> (m_numDofs, m_tmp, m_mass, v, m_savedV, m_gravity, m_dt);
	Scalar inertia;
	CublasCaller<Scalar>::sum(m_cublasHandle, m_numDofs, m_tmp, &inertia);

	return 0.5 * inertia + elasticity;
}

void StepperManager::postStep()
{
	CublasCaller<Scalar>::axpy(m_cublasHandle, m_numDofs, &m_dt, m_v, m_x);
	cudaMemcpy(m_tangents, m_dynamics->getTangents(), 3 * m_numEdges * sizeof(Scalar), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_referenceFrames1, m_dynamics->getReferenceFrames1(), 3 * m_numEdges * sizeof(Scalar), cudaMemcpyDeviceToDevice);
}

void StepperManager::printTiming() const
{
	const StepperTiming& cur_timing = m_timings.back();
	std::cout << "PR: " << cur_timing.prepare << " DY: " << cur_timing.dynamics << " ("
		<< "CONS: " << cur_timing.construct
		<< "  SOLVE: " << cur_timing.solveLinear
		<< "  LS: " << cur_timing.lineSearch
		<< "  CHECK: " << cur_timing.check
		<< ") PD: " << cur_timing.post << std::endl;
	std::cout << cur_timing.derivative << std::endl;
}