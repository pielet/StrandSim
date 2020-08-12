#ifndef STRAND_STATE_H
#define STRAND_STATE_H

#include <cuda_runtime.h>
#include "../Control/Parameters.h"
#include "../Utils/CUDAMathDef.fwd.h"

class ElasticStrand;

struct StrandState
{
	StrandState(int num_vertices, cudaStream_t stream);
	~StrandState();

	//! x must be set outside before computing dependent variables
	void update(const StrandState* lastState = NULL);
	void initReferenceFrame();

	int m_numVertices;
	int m_numEdges;
	cudaStream_t m_stream;

	/* Device pointer */
	Scalar* m_x;

	Vec3x* m_edges;
	Scalar* m_lengths;
	Vec3x* m_tangents;
	Vec3x* m_referenceFrames1;
	Vec3x* m_referenceFrames2;
	Scalar* m_referenceTwists;
	Scalar* m_twists;
	Vec3x* m_materialFrames1;
	Vec3x* m_materialFrames2;
	Vec3x* m_curvatureBinormals;
	Vec4x* m_kappas;
};

#endif // !STRAND_STATE_H
