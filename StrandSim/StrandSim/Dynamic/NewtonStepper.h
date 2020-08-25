#ifndef NEWTON_STEPPER_H
#define NEWTON_STEPPER_H

#include <iostream>
#include "../Utils/Timer.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "../Control/Parameters.h"
#include "StrandForces.h"

class ElasticStrand;
struct SimulationParameters;

struct StepperTiming
{
	//double stretching_grad;
	//double stretching_hess;
	//double twisting_grad;
	//double twisting_hess;
	//double bending_grad;
	//double bending_hess;
	double construct;
	double solveLinear;
	double lineSearch;
	double check;

	StepperTiming() : construct(0), solveLinear(0), lineSearch(0), check(0) {}

	void reset()
	{
		construct = 0;
		solveLinear = 0;
		lineSearch = 0;
		check = 0;
	}

	double sum() const
	{
		return construct + solveLinear + lineSearch + check;
	}

	const StepperTiming& operator+=(const StepperTiming& other)
	{
		construct += other.construct;
		solveLinear += other.solveLinear;
		lineSearch += other.lineSearch;
		check += other.check;
		return *this;
	}

	const StepperTiming& operator/=(int time)
	{
		construct /= time;
		solveLinear /= time;
		lineSearch /= time;
		check /= time;
		return *this;
	}

	void print() const
	{
		std::cout << "CONS: " << construct 
			<< "  SOLVE: " << solveLinear 
			<< "  LS: " << lineSearch 
			<< "  CHECK: " << check 
			<< "  SUM: " << sum() << std::endl;
	}
};

class NewtonStepper
{
public:
	NewtonStepper(int index, ElasticStrand* strand, const SimulationParameters* params, cudaStream_t stream);
	~NewtonStepper();

	void prepareStep(Scalar dt);
	bool performOneIteration();
	void commitVelocities();

	const StepperTiming& getTiming() { return m_timing; }

	void setGlobalIndex(int idx) { m_globalIndex = idx; }
	int getGlobalIndex() const { return m_globalIndex; }

	Scalar* velocities() { return m_velocities; }
	const Scalar* velocities() const { return m_velocities; }

	ElasticStrand* getStrand() { return m_strand; }

protected:
	Scalar evaluateObjectValue(const Scalar* vel);
	Scalar lineSearch(const Scalar* current_v, const Scalar* gradient_dir, const Scalar* descent_dir);

	int m_globalIndex;
	int m_dof;  //< Number of dof

	cudaStream_t m_stream;
	ElasticStrand* m_strand; 
	const SimulationParameters* m_params;

	StepperTiming m_timing;
	EventTimer m_timer;

	Scalar m_dt;
	Scalar m_alpha;
	bool m_SPD;

	/* Device pointer */
	Scalar* m_mass;
	Scalar* m_gravity;

	Scalar* m_velocities;
	Scalar* m_savedVelocities;

	Scalar* m_descentDir;
	Scalar* m_tmpValue;

	StrandForces m_dynamics; // device pointer wrapper
};

#endif // !NEWTON_STEPPER_H
