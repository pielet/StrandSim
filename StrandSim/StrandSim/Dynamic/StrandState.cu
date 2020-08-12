#include "StrandState.h"
#include "../Utils/CUDAMathDef.h"
#include "../Utils/EigenDef.h"
#include "ElasticStrand.h"
#include <iostream>

__global__ void computeEdge(Vec3x* edges, Scalar* lengths, Vec3x* tangents, const Scalar* dofs)
{
	int i = threadIdx.x;

	edges[i](0) = dofs[4 * i + 4] - dofs[4 * i];
	edges[i](1) = dofs[4 * i + 5] - dofs[4 * i + 1];
	edges[i](2) = dofs[4 * i + 6] - dofs[4 * i + 2];

	lengths[i] = edges[i].norm();

	tangents[i] = edges[i] / lengths[i];
}

__global__ void computeReferenceFrame(Vec3x* refFrames1, Vec3x* refFrames2, const Vec3x* tangents,
	const Vec3x* last_tangents, const Vec3x* last_refFrames1)
{
	int i = threadIdx.x;

	refFrames1[i] = orthonormalParallelTransport(last_refFrames1[i], last_tangents[i], tangents[i]);
	refFrames2[i] = tangents[i].cross(refFrames1[i]);
}

__global__ void computeTwist(Scalar* refTwists, Scalar* twists, const Vec3x* refFrames1, const Vec3x* tangents, const Scalar* x)
{
	int i = threadIdx.x;

	Vec3x u = orthonormalParallelTransport(refFrames1[i], tangents[i], tangents[i + 1]);
	refTwists[i] = signedAngle(u, refFrames1[i + 1], tangents[i + 1]);
	twists[i] = refTwists[i] + x[4 * i + 7] - x[4 * i + 3];
}

__global__ void computeMaterialFrame(Vec3x* matFrames1, Vec3x* matFrames2, const Vec3x* refFrames1, 
	const Vec3x* refFrames2, const Scalar* dof)
{
	int i = threadIdx.x;

	const Vec3x& u = refFrames1[i];
	const Vec3x& v = refFrames2[i];
	Scalar s = sin(dof[4 * i + 3]);
	Scalar c = cos(dof[4 * i + 3]);

	matFrames1[i] = c * u + s * v;
	matFrames2[i] = -s * u + c * v;
}

__global__ void computeCurvatureBinormal(Vec3x* curvatureBinormals, const Vec3x* tangents)
{
	int i = threadIdx.x;

	const Vec3x& t0 = tangents[i];
	const Vec3x& t1 = tangents[i + 1];

	Scalar denominator = MAX(1e-12, 1.0 + t0.dot(t1));
	curvatureBinormals[i] = 2.0 * t0.cross(t1) / denominator;
}

__global__ void computeKappa(Vec4x* kappas, const Vec3x* curvatureBinormals, const Vec3x* matFrames1, const Vec3x* matFrames2)
{
	int i = threadIdx.x;

	const Vec3x& kb = curvatureBinormals[i];
	const Vec3x& m11 = matFrames1[i];
	const Vec3x& m12 = matFrames2[i];
	const Vec3x& m21 = matFrames1[i + 1];
	const Vec3x& m22 = matFrames2[i + 1];

	kappas[i] = Vec4x(kb.dot(m12), -kb.dot(m11), kb.dot(m22), -kb.dot(m21));
}

StrandState::StrandState(int num_vertices, cudaStream_t stream):
	m_numVertices(num_vertices),
	m_numEdges(num_vertices - 1),
	m_stream(stream)
{
	// Allocate memory
	cudaMalloc((void**)& m_x, (4 * num_vertices - 1) * sizeof(Scalar));
	cudaMalloc((void**)& m_edges, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_lengths, m_numEdges * sizeof(Scalar));
	cudaMalloc((void**)& m_tangents, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_referenceFrames1, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_referenceFrames2, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_referenceTwists, (m_numVertices - 2) * sizeof(Scalar));
	cudaMalloc((void**)& m_twists, (m_numVertices - 2) * sizeof(Scalar));
	cudaMalloc((void**)& m_materialFrames1, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_materialFrames2, m_numEdges * sizeof(Vec3x));
	cudaMalloc((void**)& m_curvatureBinormals, (m_numVertices - 2) * sizeof(Vec3x));
	cudaMalloc((void**)& m_kappas, (m_numVertices - 2) * sizeof(Vec4x));
}

StrandState::~StrandState()
{
	cudaFree(m_x);
	cudaFree(m_edges);
	cudaFree(m_lengths);
	cudaFree(m_tangents);
	cudaFree(m_referenceFrames1);
	cudaFree(m_referenceFrames2);
	cudaFree(m_referenceTwists);
	cudaFree(m_twists);
	cudaFree(m_materialFrames1);
	cudaFree(m_materialFrames2);
	cudaFree(m_curvatureBinormals);
	cudaFree(m_kappas);
}

void StrandState::update(const StrandState* lastState)
{
	computeEdge <<< 1, m_numEdges, 0, m_stream >>> (m_edges, m_lengths, m_tangents, m_x);
	if (lastState) {
		computeReferenceFrame <<< 1, m_numEdges, 0, m_stream >>> (m_referenceFrames1, m_referenceFrames2,
			m_tangents, lastState->m_tangents, lastState->m_referenceFrames1);
	}
	else {
		initReferenceFrame();
	}
	computeTwist <<< 1, m_numVertices - 2, 0, m_stream >>> (m_referenceTwists, m_twists, m_referenceFrames1, m_tangents, m_x);
	computeMaterialFrame <<< 1, m_numEdges, 0, m_stream >>> (m_materialFrames1, m_materialFrames2, m_referenceFrames1, m_referenceFrames2, m_x);
	computeCurvatureBinormal <<< 1, m_numVertices - 2, 0, m_stream >>> (m_curvatureBinormals, m_tangents);
	computeKappa <<< 1, m_numVertices - 2, 0, m_stream >>> (m_kappas, m_curvatureBinormals, m_materialFrames1, m_materialFrames2);
}

Eigen::Vec3x orthonormalParallelTransport(const Eigen::Vec3x& u, const Eigen::Vec3x& t0, const Eigen::Vec3x& t1)
{
	Eigen::Vec3x b = t0.cross(t1);
	Scalar bNorm = b.norm();
	if (bNorm < 1e-12)
		return u;
	b = b / bNorm;

	Eigen::Vec3x n0 = t0.cross(b);
	Eigen::Vec3x n1 = t1.cross(b);

	return u.dot(n0) * n1 + u.dot(b) * b;
}

void StrandState::initReferenceFrame()
{
	Eigen::VecXx refFrames1(3 * m_numEdges);
	Eigen::VecXx refFrames2(3 * m_numEdges);
	Eigen::VecXx tangents(3 * m_numEdges);
	cudaMemcpyAsync(tangents.data(), m_tangents, 3 * m_numEdges * sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	// Computes reference frames of the first edge
	Eigen::Vec3x u = Eigen::Vec3x::Random();
	Eigen::Vec3x t = tangents.segment<3>(0);
	refFrames1.segment<3>(0) = (u - u.dot(t) * t).normalized();
	refFrames2.segment<3>(0) = t.cross(refFrames1.segment<3>(0));

	for (int i = 1; i < m_numEdges; ++i)
	{
		Eigen::Vec3x t0 = tangents.segment<3>(3 * i - 3);
		Eigen::Vec3x t1 = tangents.segment<3>(3 * i);
		u = refFrames1.segment<3>(3 * i - 3);

		u = orthonormalParallelTransport(u, t0, t1);
		refFrames1.segment<3>(3 * i) = u;
		refFrames2.segment<3>(3 * i) = t1.cross(u);
	}

	cudaMemcpyAsync(m_referenceFrames1, refFrames1.data(), 3 * m_numEdges * sizeof(Scalar), cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_referenceFrames2, refFrames2.data(), 3 * m_numEdges * sizeof(Scalar), cudaMemcpyHostToDevice, m_stream);
}
