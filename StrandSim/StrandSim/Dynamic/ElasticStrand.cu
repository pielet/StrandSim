#include "ElasticStrand.h"
#include "../Utils/CUDAMathDef.h"

__global__ void computeLength(Scalar* lengths, const Scalar* x)
{
	int i = threadIdx.x;

	Vec3x x0(x[4 * i], x[4 * i + 1], x[4 * i + 2]);
	Vec3x x1(x[4 * i + 4], x[4 * i + 5], x[4 * i + 6]);

	lengths[i] = (x1 - x0).norm();
}

__global__ void computeVoronoiLength(Scalar* VoronoiLen, Scalar* invVoronoiLen, const Scalar* lengths, int numVertices)
{
	int i = threadIdx.x;

	VoronoiLen[i] = 0;
	if (i < numVertices - 1)
	{
		VoronoiLen[i] += 0.5 * lengths[i];
	}
	if (i > 0)
	{
		VoronoiLen[i] += 0.5 * lengths[i - 1];
	}
	invVoronoiLen[i] = 1. / VoronoiLen[i];
}

__global__ void computeVertexMass(Scalar* vtxMass, const Scalar* VoronoiLen, Scalar mass_per_length)
{
	int i = threadIdx.x;

	vtxMass[i] = mass_per_length * VoronoiLen[i];
}

__global__ void computeEdgeMass(Scalar* edgeMass, Scalar* edgeInertia, const Scalar* lengths, Scalar radius, Scalar mass_per_length)
{
	int i = threadIdx.x;

	edgeMass[i] = mass_per_length * lengths[i];
	edgeInertia[i] = 0.5 * edgeMass[i] * radius * radius;
}

ElasticStrand::ElasticStrand(int globalIdx, int dof, const Scalar* x, const FixedMap& fixed, const StrandParameters* params, cudaStream_t stream):
	m_globalIdx(globalIdx),
	m_numVertices(dof / 4 + 1),
	m_numEdges(dof / 4),
	m_numDof(dof),
	m_fixed(fixed),
	m_parameters(params),
	m_stream(stream),
	m_currentState(dof / 4 + 1, stream)
{
	cudaMalloc((void**)& m_restLengths, m_numEdges * sizeof(Scalar));
	cudaMalloc((void**)& m_VoronoiLengths, m_numVertices * sizeof(Scalar));
	cudaMalloc((void**)& m_invVoronoiLengths, m_numVertices * sizeof(Scalar));
	cudaMalloc((void**)& m_restTwists, (m_numVertices - 2) * sizeof(Scalar));
	cudaMalloc((void**)& m_restKappas, (m_numVertices - 2) * sizeof(Vec4x));
	cudaMalloc((void**)& m_vertexMasses, m_numVertices * sizeof(Scalar));
	cudaMalloc((void**)& m_edgeMasses, m_numEdges * sizeof(Scalar));
	cudaMalloc((void**)& m_edgeInertia, m_numEdges * sizeof(Scalar));

	cudaMalloc((void**)& m_fixedIndices, m_numVertices * sizeof(int));
	cudaMalloc((void**)& m_fixedPositions, m_numVertices * sizeof(Vec3x));

	// Copy data
	cudaMemcpyAsync(m_currentState.m_x, x, dof * sizeof(Scalar), cudaMemcpyHostToDevice, m_stream);
	updateFixedVector();

	// Compute initial state
	m_currentState.update();

	// Compute dependent data
	assign <<< 1, m_numEdges, 0, m_stream >>> (m_restLengths, m_currentState.m_lengths);
	computeVoronoiLength <<< 1, m_numVertices, 0, m_stream >>> (m_VoronoiLengths, m_invVoronoiLengths, m_restLengths, m_numVertices);
	assign <<< 1, m_numVertices - 2, 0, m_stream >>> (m_restTwists, m_currentState.m_twists);
	assign <<< 1, m_numVertices - 2, 0, m_stream >>> (m_restKappas, m_currentState.m_kappas);
	Scalar mass_per_length = M_PI * params->m_radius * params->m_radius * params->m_density;
	computeVertexMass <<< 1, m_numVertices, 0, m_stream >>> (m_vertexMasses, m_VoronoiLengths, mass_per_length);
	computeEdgeMass <<< 1, m_numEdges, 0, m_stream >>> (m_edgeMasses, m_edgeInertia, m_restLengths, params->m_radius, mass_per_length);
}

ElasticStrand::~ElasticStrand()
{
	cudaFree(m_restLengths);
	cudaFree(m_VoronoiLengths);
	cudaFree(m_invVoronoiLengths);
	cudaFree(m_restTwists);
	cudaFree(m_restKappas);
	cudaFree(m_vertexMasses);
	cudaFree(m_edgeMasses);
	cudaFree(m_edgeInertia);

	cudaFree(m_fixedIndices);
	cudaFree(m_fixedPositions);
}

void ElasticStrand::updateFixedVector()
{
	if (m_fixed.size() > 0)
	{
		int n_fixed = m_fixed.size();
		std::vector<int> indices(n_fixed);
		std::vector<Vec3x> positions(n_fixed);

		int i = 0;
		for (auto it = m_fixed.begin(); it != m_fixed.end(); ++it)
		{
			indices[i] = it->first;
			positions[i] = Vec3x(3, it->second.data());
			++i;
		}

		cudaMemcpyAsync(m_fixedIndices, indices.data(), n_fixed * sizeof(int), cudaMemcpyHostToDevice, m_stream);
		cudaMemcpyAsync(m_fixedPositions, positions.data(), n_fixed * sizeof(Vec3x), cudaMemcpyHostToDevice, m_stream);
	}
}
