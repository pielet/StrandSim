#ifndef NEWTON_STEPPER_H
#define NEWTON_STEPPER_H

#include "../Utils/Timer.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "../Control/Parameters.h"
#include "StrandForces.h"


class ElasticStrand;
struct SimulationParameters;

struct StepperTiming
{
	double hessian;
	double gradient;
	double factorize;
	double solveLinear;
	double lineSearch;

	StepperTiming() : hessian(0), gradient(0), factorize(0), solveLinear(0), lineSearch(0) {}
	StepperTiming(double h, double g, double f, double s, double l) :
		hessian(h), gradient(g), factorize(f), solveLinear(s), lineSearch(l) {}

	void reset()
	{
		hessian = 0;
		gradient = 0;
		factorize = 0;
		solveLinear = 0;
		lineSearch = 0;
	}

	double sum() const
	{
		return hessian + gradient + factorize + solveLinear + lineSearch;
	}

	const StepperTiming& operator+=(const StepperTiming& other)
	{
		hessian += other.hessian;
		gradient += other.gradient;
		factorize += other.factorize;
		solveLinear += other.factorize;
		lineSearch += other.lineSearch;
		return *this;
	}

	const StepperTiming& operator/=(int time)
	{
		hessian /= time;
		gradient /= time;
		factorize /= time;
		solveLinear /= time;
		lineSearch /= time;
		return *this;
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
	void outputTiming() const;

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
