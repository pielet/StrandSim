#ifndef STEPPER_MANAGER_H
#define STEPPER_MANAGER_H

#include <iostream>
#include <vector>
#include "../Utils/Timer.h"
#include "../Utils/Cublas.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "BandMatrix.h"
#include "LinearSolver.h"
#include "StrandStates.h"

class StepperManager
{
public:
	StepperManager(Scalar dt, int n_dof, int n_fixed, const Scalar* x, const Scalar* v, const std::vector<int>& strand_ptr, 
		const int* fixed_idx, const Scalar* fixed_pos, const SimulationParameters* sim_params, 
		const StrandParameters* strand_params, const CollisionParameters* col_params);
	~StepperManager();

	int getNumDofs() const { return m_numDofs; }
	int getNumStrands() const { return m_numStrands; }
	int getNumVertices() const { return m_numVertices; }
	int getNumEdges() const { return m_numEdges; }

	const int* getStrandPtr() const { return m_strand_ptr; }
	const int* getVtxToStrand() const { return m_vtx_2_strand; }

	const Scalar* getX() const { return m_x; }

	const Vec3x* getTangents() const { return m_tangents; }
	const Vec3x* getMaterialFrames1() const{ return m_dynamics->getMaterialFrames1(); }

	void setTime(Scalar time) { m_time = time; }

	void step();

protected:
	struct StepperTiming
	{
		double prepare;
		double dynamics;
		double post;

		double construct;
		double solveLinear;
		double lineSearch;
		double check;

		DerivativeTiming derivative;

		StepperTiming() : prepare(0), dynamics(0.), post(0), construct(0), solveLinear(0), lineSearch(0), check(0) {}

		void reset()
		{
			prepare = 0.;
			dynamics = 0.;
			post = 0.;
			construct = 0.;
			solveLinear = 0.;
			lineSearch = 0.;
			check = 0.;
			derivative.reset();
		}

		double dynamicsSum() const
		{
			return construct + solveLinear + lineSearch + check;
		}
	};

	void printTiming() const;

	void prepareStep();
	bool performNewtonStep();
	void postStep();

	Scalar evaluateObjectValue(const Scalar* v);
	Scalar lineSearch(const Scalar* current_v, const Scalar* gradient_dir, const Scalar* descent_dir);

	Scalar m_time;  //< Current time
	Scalar m_dt;    //< Timer per frame

	int m_numDofs;
	int m_numStrands;
	int m_numVertices;
	int m_numEdges;
	int m_numInnerVtx;
	int m_numFixed;

	int m_maxVtxPerStrand;

	int m_dofBlocks;

	Scalar m_lsStep;

	std::vector<StepperTiming> m_timings;
	StepperTiming m_timing;
	EventTimer m_timer;

	const SimulationParameters* m_simParams;
	const StrandParameters* m_strandParams;
	const CollisionParameters* m_colParams;

	cublasHandle_t m_cublasHandle;

	/* Device pointers */
	int* m_fixed_idx;
	int* m_strand_ptr;
	int* m_vtx_2_strand;
	int* m_edge_2_strand;
	int* m_inner_vtx_2_strand;

	Scalar* m_x;
	Scalar* m_v;
	Scalar* m_savedV;
	Scalar* m_mass;
	Scalar* m_gravity;
	Vec3x* m_referenceFrames1;
	Vec3x* m_materialFrames1;
	Vec3x* m_tangents;
	Vec3x* m_fixedTargets;

	Scalar* m_tmp;
	Scalar* m_descentDir;

	HessianMatrix* m_A;
	LinearSolver* m_linearSolver;
	StrandStates* m_dynamics;
};

#endif // !STEPPER_MANAGER_H
