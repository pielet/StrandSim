#ifndef ELASTIC_STRAND_H
#define ELASTIC_STRAND_H

#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "../Control/Parameters.h"
#include "../Utils/EigenDef.h"
#include "../Utils/CUDAMathDef.fwd.h"
#include "StrandState.h"

typedef std::map<int, Eigen::Vec3x> FixedMap;

class ElasticStrand
{
public:
	ElasticStrand(int globalIdx, int dof, const Scalar* x, const FixedMap& fixed,
		const StrandParameters* params, cudaStream_t stream = 0);
	~ElasticStrand();

	int getGlobalIndex() const { return m_globalIdx; }
	int getNumVertices() const { return m_numVertices; }
	int getNumEdges() const { return m_numEdges; }
	int getNumDof() const { return m_numDof; }

	const StrandParameters* getParameters() const { return m_parameters; }

	cudaStream_t getStream() const { return m_stream; }

	const StrandState* getCurrentState() const { return &m_currentState; }
	void updateCurrentState() { m_currentState.update(&m_currentState); }

	FixedMap& getFixedMap() { return m_fixed; }
	const FixedMap& getFixedMap() const { return m_fixed; }
	int getNumFixed() const { return m_fixed.size(); }
	void updateFixedVector();

	Scalar* getX() { return m_currentState.m_x; }
	const Scalar* getX() const { return m_currentState.m_x; }
	const Scalar* getRestLengths() const { return m_restLengths; }
	const Scalar* getInvVtxLengths() const { return m_invVoronoiLengths; }
	const Scalar* getRestTwists() const { return m_restTwists; }
	const Vec4x* getRestKappas() const { return m_restKappas; }
	const Scalar* getVertexMasses() const { return m_vertexMasses; }
	const Scalar* getEdgeInertia() const { return m_edgeInertia; }

	const int* getFixedIndices() const { return m_fixedIndices; }
	const Vec3x* getFixedPositions() const { return m_fixedPositions; }

protected:
	int m_globalIdx;
	int m_numVertices;
	int m_numEdges;
	int m_numDof;

	FixedMap m_fixed;

	const StrandParameters* m_parameters;

	mutable cudaStream_t m_stream;

	/* Device pointer */
	StrandState m_currentState; // device pointer wrapper

	// Rest pose
	Scalar* m_restLengths;
	Scalar* m_VoronoiLengths;
	Scalar* m_invVoronoiLengths;
	Scalar* m_restTwists;
	Vec4x* m_restKappas;

	Scalar* m_vertexMasses;
	Scalar* m_edgeMasses;
	Scalar* m_edgeInertia;

	int* m_fixedIndices;
	Vec3x* m_fixedPositions;
};

#endif // !ELASTIC_STRAND_H
