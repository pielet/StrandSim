#ifndef STRAND_STATES_H
#define STRAND_STATES_H

#include <iostream>
#include "../Utils/Cublas.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "../Utils/Timer.h"
#include "BandMatrix.h"

#include "../Utils/EigenDef.h"

struct DerivativeTiming
{
	double stretchingGradient;
	double stretchingHessian;
	double twistingGradient;
	double twistingHessian;
	double bendingGradient;
	double bendingHessian;
	double fixingGradient;
	double fixingHessian;

	DerivativeTiming() :stretchingGradient(0.), stretchingHessian(0.), twistingGradient(0.), 
		twistingHessian(0.), bendingGradient(0.), bendingHessian(0.), fixingGradient(0.), fixingHessian(0.) {}

	void reset()
	{
		stretchingGradient = 0.;
		stretchingHessian = 0.;
		twistingGradient = 0.;
		twistingHessian = 0.;
		bendingGradient = 0.;
		bendingHessian = 0.;
		fixingGradient = 0.;
		fixingHessian = 0.;
	}

	DerivativeTiming& operator+=(const DerivativeTiming& other)
	{
		stretchingGradient += other.stretchingGradient;
		stretchingHessian += other.stretchingHessian;
		twistingGradient += other.twistingGradient;
		twistingHessian += other.twistingHessian;
		bendingGradient += other.bendingGradient;
		bendingHessian += other.bendingHessian;
		fixingGradient += other.fixingGradient;
		fixingHessian += other.fixingHessian;

		return *this;
	}

	DerivativeTiming& operator/=(int time)
	{
		stretchingGradient /= time;
		stretchingHessian /= time;
		twistingGradient /= time;
		twistingHessian /= time;
		bendingGradient /= time;
		bendingHessian /= time;
		fixingGradient /= time;
		fixingHessian /= time;

		return *this;
	}

	friend std::ostream& operator<<(std::ostream& os, const DerivativeTiming& dt)
	{
		os << "SG: " << dt.stretchingGradient << " SH: " << dt.stretchingHessian
			<< " TG: " << dt.twistingGradient << " TH: " << dt.twistingHessian
			<< " BG: " << dt.bendingGradient << " BH: " << dt.bendingHessian
			<< " FG: " << dt.fixingGradient << " FH: " << dt.fixingHessian;
		return os;
	}
};


class StrandStates
{
public:
	StrandStates(const StrandParameters* params, int num_dofs, int num_strands, int num_vertices, int max_vtx, int num_fixed, int* fixed_idx, const int* strand_ptr, const int* vtx_to_strand, 
		const int* edge_to_strand, const int* inner_vtx_to_strand, Vec3x* fixed_targets, HessianMatrix* hessian);
	~StrandStates();

	const DerivativeTiming& getTiming() const { return m_timing; }
	
	Scalar* getX() { return m_x; }

	Scalar* getMass() { return m_mass; }
	Scalar* getGravity() { return m_gravity; }

	const Vec3x* getReferenceFrames1() const { return m_referenceFrames1; }
	const Vec3x* getMaterialFrames1() const { return m_materialFrames1; }
	const Vec3x* getTangents() const { return m_tangents; }

	Scalar* getTotalForces() const { return m_totalForces; }

	//! m_x must be set outside before calling these functions
	void init();
	void updateStates(const Vec3x* last_tangents, const Vec3x* last_ref1);
	void computeForcesAndJacobian(bool, bool withStretch = true, bool withTwist = true, bool withBend = true);
	Scalar computeEnergy();

protected:
	const StrandParameters* m_params;
	cublasHandle_t m_cublasHandle;
	DerivativeTiming m_timing;
	EventTimer m_timer;

	int m_numDofs;
	int m_numStrands;
	int m_numVertices;
	int m_numEdges;
	int m_numInnerVtx;
	int m_numFixed;

	int m_maxVtxPerStrand;

	int strand_blocksPerGrid;
	int vtx_blocksPerGrid;
	int edge_blocksPerGrid;
	int ivtx_blocksPerGrid;
	int fixed_blocksPerGrid;

	/* Data on device */
	// Indices
	int* m_fixed_idx;
	const int* m_strand_ptr;
	const int* m_vertex_to_strand;
	const int* m_edge_to_strand;
	const int* m_inner_vtx_to_strand;

	// Inital states
	Scalar* m_mass;
	Scalar* m_gravity;
	Scalar* m_invVtxLengths;
	Scalar* m_restLengths;
	Scalar* m_restTwists;
	Vec4x* m_restKappas;

	Vec3x* m_fixedTargets;

	// States
	Scalar* m_x;
	Scalar* m_lengths;
	Vec3x* m_tangents;
	Vec3x* m_referenceFrames1;
	Scalar* m_twists;
	Vec3x* m_materialFrames1;
	Vec3x* m_materialFrames2;
	Vec3x* m_curvatureBinormals;
	Vec4x* m_kappas;

	// Energies
	Scalar* m_fixingEnergy;
	Scalar* m_stretchingEnergy;
	Scalar* m_twistingEnergy;
	Scalar* m_bendingEnergy;

	// Forces
	Scalar* m_totalForces;
	Vec11x* m_gradTwists;
	Mat11x4x* m_gradKappas;

	// Jacobian
	HessianMatrix* m_totalJacobian;	//< device pointer wrapper
	Mat11x* m_hessTwists;
	Mat11x* m_hessKappas;
};

#endif // !STRAND_STATES_H
