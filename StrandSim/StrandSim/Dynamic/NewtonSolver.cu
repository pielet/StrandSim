#include "NewtonStepper.h"

#include <iostream>
#include "../Utils/CUDAMathDef.h"
#include "BandMatrix.h"
#include "StrandState.h"
#include "ElasticStrand.h"
#include "StrandForces.h"

__global__ void initMass(Scalar* mass, const Scalar* vtxMass, const Scalar* edgeInertia)
{
	int i = threadIdx.x;

	mass[4 * i] = mass[4 * i + 1] = mass[4 * i + 2] = vtxMass[i];
	if (i < blockDim.x)
		mass[4 * i + 3] = edgeInertia[i];
}

__global__ void initGravity(Scalar* gravity, const Scalar* mass, Scalar gx, Scalar gy, Scalar gz)
{
	int i = threadIdx.x;

	gravity[4 * i] = gx * mass[4 * i];
	gravity[4 * i + 1] = gy * mass[4 * i + 1];
	gravity[4 * i + 2] = gz * mass[4 * i + 2];

	if (4 * i + 3 < blockDim.x)
		gravity[4 * i + 3] = 0.;
}

__global__ void computeGradient(Scalar* forces, const Scalar* mass, const Scalar* next_v, const Scalar* v, 
	const Scalar* gravity, Scalar dt)
{
	int i = threadIdx.x;
	forces[i] = mass[i] * (next_v[i] - v[i]) - dt * (gravity[i] + forces[i]);
}

__global__ void computeInertiaTerm(Scalar* inertia, const Scalar* mass, const Scalar* next_v, const Scalar* v,
	const Scalar* gravity, Scalar dt)
{
	int i = threadIdx.x;
	inertia[i] = next_v[i] - v[i] - dt * gravity[i] / mass[i];
}

NewtonStepper::NewtonStepper(int index, ElasticStrand* strand, const SimulationParameters* params, cudaStream_t stream) :
	m_globalIndex(index),
	m_dof(strand->getNumDof()),
	m_strand(strand),
	m_params(params),
	m_stream(stream),
	m_SPD(true),
	m_alpha(1.),
	m_dynamics(strand, stream)
{
	// Allocate memory
	cudaMalloc((void**)& m_mass, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& m_gravity, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& m_velocities, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& m_savedVelocities, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& m_descentDir, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& m_tmpValue, m_dof * sizeof(Scalar));

	cudaMalloc((void**)& x_k, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& r_k, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& Ap, m_dof * sizeof(Scalar));
	cudaMalloc((void**)& p_k, m_dof * sizeof(Scalar));

	// Init v0, mass and gravity
	set<Scalar> <<< 1, m_dof, 0, stream >>> (m_dof, m_velocities, 0.0);
	initMass <<< 1, strand->getNumVertices(), 0, stream >>> (m_mass, strand->getVertexMasses(), strand->getEdgeInertia());
	initGravity <<< 1, strand->getNumVertices(), 0, stream >>> (m_gravity, m_mass, 0., -981.0, 0.);
}

NewtonStepper::~NewtonStepper()
{
	cudaFree(m_mass);
	cudaFree(m_gravity);
	cudaFree(m_velocities);
	cudaFree(m_savedVelocities);
	cudaFree(m_descentDir);
	cudaFree(m_tmpValue);

	cudaFree(x_k);
	cudaFree(r_k);
	cudaFree(Ap);
	cudaFree(p_k);
}

void NewtonStepper::prepareStep(Scalar dt)
{
	m_dt = dt;
	m_alpha = 1.0;
	m_timing.reset();
	assign <<< 1, m_dof, 0, m_stream >>> (m_savedVelocities, m_velocities);
}

bool NewtonStepper::performOneIteration()
{
	//Eigen::VecXx test(m_dof);
	Timer tt;

	// Compute future forces and gradient
	m_timer.start(m_stream);
	add <<< 1, m_dof, 0, m_stream >>> (m_dynamics.getX(), m_strand->getX(), m_velocities, m_dt);
	//cudaMemcpy(test.data(), m_dynamics.getX(), m_dof * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "new x: \n" << test << std::endl;
	m_dynamics.computeForceAndGradient();

	//cudaMemcpy(test.data(), m_strand->getX(), m_dof * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "saved x: \n" << test << std::endl;
	//cudaMemcpy(test.data(), m_velocities, m_dof * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//std::cout << "v: \n" << test << std::endl;

	// Compute objective function's gradient and hassian
	Scalar* gradient = m_dynamics.getTotalForce();
	computeGradient <<< 1, m_dof, 0, m_stream >>> (gradient, m_mass, m_velocities, m_savedVelocities, m_gravity, m_dt);
	HessianMatrix& hessian = m_dynamics.getHessian();
	hessian.multiplyInPlace(m_dt * m_dt);
	hessian.addInDiagonal(m_mass);
	m_timing.construct += m_timer.elapsedMilliseconds();

	// Solve linear equation
	m_timer.start(m_stream);
	//m_SPD = CUDASolveChol<Scalar>(&hessian, gradient, m_descentDir);
	//if (!m_SPD)
	//{
	//	std::cerr << "NewtonStepper " << m_globalIndex << " is NOT SPD" << std::endl;
	//	exit(-1);
	//}
	//assign <<< 1, m_dof, 0, m_stream >>> (m_descentDir, m_descentDir, -1.);
	solveLinearSys(&hessian, gradient);
	//solveLinearSysKernel <<< 1, 1, 0, m_stream >>> (m_dof, hessian.getMatrix(), gradient, x_k, r_k, p_k, Ap);
	assign <<< 1, m_dof, 0, m_stream >>> (m_descentDir, x_k, -1.);
	m_timing.solveLinear += m_timer.elapsedMilliseconds();

	// Line-search
	m_timer.start(m_stream);
	Scalar obj_value = lineSearch(m_velocities, gradient, m_descentDir);
	add <<< 1, m_dof, 0, m_stream >>> (m_velocities, m_velocities, m_descentDir, m_alpha);
	m_timing.lineSearch += m_timer.elapsedMilliseconds();

	// Check convergence condition
	tt.start();
	Scalar delta_v;
	dot <<< 1, m_dof, m_dof * sizeof(Scalar), m_stream >>> (m_tmpValue, m_descentDir, m_descentDir);
	cudaMemcpyAsync(&delta_v, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	delta_v = m_alpha * sqrt(delta_v);
	//std::cout << "Step size: " << m_alpha << "  Residual: " << obj_value << " Delta v: " << delta_v << std::endl;
	m_timing.check += tt.elapsedMilliseconds();

	if (delta_v / m_dof < m_params->m_newtonTolerance)
		return true;
	return false;
}

void NewtonStepper::commitVelocities()
{
	add <<< 1, m_dof, 0, m_stream >>> (m_strand->getX(), m_strand->getX(), m_velocities, m_dt);
	m_strand->updateCurrentState();
}

Scalar NewtonStepper::lineSearch(const Scalar* current_v, const Scalar* gradient_dir, const Scalar* descent_dir)
{
	Scalar current_obj_value = evaluateObjectValue(current_v);

	if (m_params->m_useLineSearch)
	{
		Scalar next_obj_value, rhs;
		Scalar grad_dot_desc;
		dot << < 1, m_dof, m_dof * sizeof(Scalar), m_stream >> > (m_tmpValue, gradient_dir, descent_dir);
		cudaMemcpyAsync(&grad_dot_desc, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);

		m_alpha = std::min(1., 2 * std::max(1e-5, m_alpha)) / m_params->m_lsBeta;

		do {
			m_alpha *= m_params->m_lsBeta;
			if (m_alpha < 1e-5)
				break;
			add << < 1, m_dof, 0, m_stream >> > (m_tmpValue, current_v, descent_dir, m_alpha);
			next_obj_value = evaluateObjectValue(m_tmpValue);
			rhs = current_obj_value + m_params->m_lsAlpha * m_alpha * grad_dot_desc;
		} while (next_obj_value > rhs);

		return next_obj_value;
	}
	else return current_obj_value;
}

Scalar NewtonStepper::evaluateObjectValue(const Scalar* vel)
{
	add <<< 1, m_dof, 0, m_stream >>> (m_dynamics.getX(), m_strand->getX(), vel, m_dt);
	m_dynamics.computeEnergy();
	Scalar elasticity = m_dynamics.getEnergy();

	computeInertiaTerm <<< 1, m_dof, 0, m_stream >>> (m_tmpValue, m_mass, vel, m_savedVelocities, m_gravity, m_dt);
	dot <<< 1, m_dof, m_dof * sizeof(Scalar), m_stream >>> (m_tmpValue, m_tmpValue, m_tmpValue, m_mass);
	Scalar inertia;
	cudaMemcpyAsync(&inertia, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	return 0.5 * inertia + elasticity;
}

void NewtonStepper::solveLinearSys(const HessianMatrix* A, const Scalar* b)
{
	EventTimer tt;

	tt.start();
	set <<< 1, m_dof, 0, m_stream >>> (m_dof, x_k, 0.);
	A->multiplyVec(Ap, x_k);
	add <<< 1, m_dof, 0, m_stream >>> (r_k, b, Ap, -1.0);
	assign <<< 1, m_dof, 0, m_stream >>> (p_k, r_k);

	Scalar rs_old, rs_new, alpha;
	dot <<< 1, m_dof, m_dof * sizeof(Scalar), m_stream >>> (m_tmpValue, r_k, r_k);
	cudaMemcpyAsync(&rs_old, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	m_timing.init += tt.elapsedMilliseconds();

	for (int k = 0; k < m_dof; ++k)
	{
		tt.start();
		A->multiplyVec(Ap, p_k);
		m_timing.multiply += tt.elapsedMilliseconds();

		tt.start();
		dot <<< 1, m_dof, m_dof * sizeof(Scalar), m_stream >>> (m_tmpValue, p_k, Ap);
		cudaMemcpyAsync(&alpha, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);
		m_timing.dot += tt.elapsedMilliseconds();

		tt.start();
		alpha = rs_old / alpha;
		add <<< 1, m_dof, 0, m_stream >>> (x_k, x_k, p_k, alpha);
		add <<< 1, m_dof, 0, m_stream >>> (r_k, r_k, Ap, -alpha);
		m_timing.add += tt.elapsedMilliseconds();

		tt.start();
		dot <<< 1, m_dof, m_dof * sizeof(Scalar), m_stream >>> (m_tmpValue, r_k, r_k);
		cudaMemcpyAsync(&rs_new, m_tmpValue, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);
		m_timing.dot += tt.elapsedMilliseconds();

		std::cout << "residual: " << sqrt(rs_new) << std::endl;

		if (sqrt(rs_new) < 1e-10) {
			//m_timing.multiply /= k + 1;
			//m_timing.dot /= k + 1;
			//m_timing.add /= k + 1;
			std::cout << "break k: " << k << std::endl;
			break;
		}

		tt.start();
		add <<< 1, m_dof, 0, m_stream >>> (p_k, r_k, p_k, rs_new / rs_old);
		rs_old = rs_new;
		m_timing.add += tt.elapsedMilliseconds();
	}
}