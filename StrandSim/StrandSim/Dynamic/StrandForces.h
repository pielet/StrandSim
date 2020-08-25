#ifndef STRAND_FORCES_H
#define STRAND_FORCES_H

#include "../Control/Parameters.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "../Utils/Timer.h"
#include "BandMatrix.h"
#include "StrandState.h"

class StrandParameters;
class ElasticStrand;

//struct ForcesTiming
//{
//	double stretching_grad;
//	double stretching_hess;
//	double twisting_grad;
//	double twisting_hess;
//	double bending_grad;
//	double bending_hess;
//};

class StrandForces
{
public:
	StrandForces(const ElasticStrand* strand, cudaStream_t stream);
	~StrandForces();

	//! x must be set outside before calling this function
	void computeForceAndGradient();
	//! x must be set outside before calling this function
	void computeEnergy();

	Scalar* getX() { return m_state.m_x; }
	Scalar getEnergy() const { return m_totalEnergy; }
	Scalar* getTotalForce() { return m_totalForces; }
	HessianMatrix& getHessian() { return m_totalGradient; }
	const HessianMatrix& getHessian() const { return m_totalGradient; }

protected:
	void computeFixing();
	void computeStretching();
	void computeTwisting();
	void computeBending();

	const ElasticStrand* m_strand;
	const StrandParameters* m_params;

	int m_numVertices;
	int m_numEdges;

	cudaStream_t m_stream;

	Scalar m_totalEnergy;

	/* Device pointers */
	StrandState m_state;  // device pointer wrapper

	Scalar* m_fixingEnergy;
	Scalar* m_stretchingEnergy;
	Scalar* m_twistingEnergy;
	Scalar* m_bendingEnergy;

	Vec11x* m_gradTwists;
	Mat11x* m_hessTwists;
	Mat11x4x* m_gradKappas;
	Mat11x* m_hessKappas;
	
	Scalar* m_totalForces;
	HessianMatrix m_totalGradient; // device pointer wrapper
};

#endif // !STRAND_FORCES_H
