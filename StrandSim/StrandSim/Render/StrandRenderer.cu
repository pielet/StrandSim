#include "StrandRenderer.h"
#include <vector>
#include <iostream>
#include <cuda_gl_interop.h>
#include "../Utils/CUDAMathDef.h"
#include "../Render/Shader.h"
#include "../Dynamic/StepperManager.h"

const int slides = 8;

__global__ void updateStrand(int num_vertices, const int* vtx_to_strand, Vec3f* strand_vtx, const Scalar* x)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_vertices) return;

	int j = 4 * i - vtx_to_strand[i];
	strand_vtx[i] = Vec3f(x[j], x[j + 1], x[j + 2]);
}

__global__ void updateQuad(int nv, int slides, const int* vtx_to_strand, const int* strand_ptr, const Vec3f* vtx, const Vec3x* tangents, const Vec3x* materialFrame1, float radius, Vec3f* quad_vtx)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nv) return;

	int sid = vtx_to_strand[i];
	int j = i - sid;

	Vec3x tangent;
	Vec3x normal;
	if (i == strand_ptr[sid + 1] - 1)
	{
		tangent = tangents[j - 1];
		normal = materialFrame1[j - 1];
	}
	else
	{
		tangent = tangents[j];
		normal = materialFrame1[j];
	}

	for (int k = 0; k < slides; ++k)
	{
		quad_vtx[slides * i + k] = vtx[i] + radius * Vec3f(normal(0), normal(1), normal(2));
		rotateAxisAngle(normal, tangent, 2.0 / slides);
	}
}

StrandRenderer::StrandRenderer(int num_vertices, int num_edges, float radius, const std::vector<int>& strand_ptr, const StepperManager* stepper) :
	m_numVertices(num_vertices),
	m_numEdges(num_edges),
	m_radius(radius),
	m_strand_ptr(strand_ptr),
	m_stepper(stepper),
	m_geometryChanged(true)
{
	// Init buffers
	glGenBuffers(1, &m_strandVerticesBuffer);
	glGenBuffers(1, &m_strandIndicesBuffer);
	glGenBuffers(1, &m_quadVerticesBuffer);
	glGenBuffers(1, &m_quadIndicesBuffer);

	// Generate vertice buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_strandVerticesBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m_quadVerticesBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * slides * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);

	// Buffer indices data
	std::vector<unsigned> strand_indices(num_edges * 2);
	std::vector<unsigned> quad_indices(num_edges * slides * 4);
	int i = 0;
	for (int sid = 0; sid < strand_ptr.size() - 1; ++sid)
	{
		for (int j = strand_ptr[sid]; j < strand_ptr[sid + 1] - 1; ++j)
		{
			strand_indices[i++] = j;
			strand_indices[i++] = j + 1;
		}
	}
	i = 0;
	for (int sid = 0; sid < strand_ptr.size() - 1; ++sid) 
	{
		for (int j = strand_ptr[sid]; j < strand_ptr[sid + 1] - 1; ++j)
		{
			int cur_idx = slides * j;
			for (int k = 0; k < slides; ++k)
			{
				int k1 = (k + 1) % slides;
				quad_indices[i++] = cur_idx + k;
				quad_indices[i++] = cur_idx + k1;
				quad_indices[i++] = cur_idx + k1 + slides;
				quad_indices[i++] = cur_idx + k + slides;
			}
		}
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_strandIndicesBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, strand_indices.size() * sizeof(unsigned), strand_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_quadIndicesBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.size() * sizeof(unsigned), quad_indices.data(), GL_STATIC_DRAW);

	// Register CUDA resources
	cudaGraphicsGLRegisterBuffer(&m_cudaStrandVerticesResource, m_strandVerticesBuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&m_cudaQuadVerticesResource, m_quadVerticesBuffer, cudaGraphicsMapFlagsWriteDiscard);
}

StrandRenderer::~StrandRenderer()
{
	cudaGraphicsUnregisterResource(m_cudaStrandVerticesResource);
	cudaGraphicsUnregisterResource(m_cudaQuadVerticesResource);

	glDeleteBuffers(1, &m_strandVerticesBuffer);
	glDeleteBuffers(1, &m_quadVerticesBuffer);
	glDeleteBuffers(1, &m_quadIndicesBuffer);
}

void StrandRenderer::render(const Shader* shader)
{
	if (m_geometryChanged)
	{
		updateVertices();
		m_geometryChanged = false;
	}

	glEnableVertexAttribArray(0);

	// Draw strand line
	glBindBuffer(GL_ARRAY_BUFFER, m_strandVerticesBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_strandIndicesBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	shader->setVec4f("vertex_color", 0.f, 0.f, 0.f, 1.f);  // black
	glDrawElements(GL_LINES, m_numEdges * 2, GL_UNSIGNED_INT, 0);

	// Draw quads
	glBindBuffer(GL_ARRAY_BUFFER, m_quadVerticesBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_quadIndicesBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	shader->setVec4f("vertex_color", 0.f, 0.f, 1.f, 0.2f);  // transparent blue
	glDrawElements(GL_QUADS, m_numEdges * slides * 4, GL_UNSIGNED_INT, 0);
}

void StrandRenderer::updateVertices()
{
	// Map data
	Vec3f* strand_vtx;
	Vec3f* quad_vtx;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &m_cudaStrandVerticesResource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& strand_vtx, &num_bytes, m_cudaStrandVerticesResource);
	cudaGraphicsMapResources(1, &m_cudaQuadVerticesResource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& quad_vtx, &num_bytes, m_cudaQuadVerticesResource);

	// Launch kernal
	int blocks_per_grid = (m_numVertices + g_threadsPerBlock - 1) / g_threadsPerBlock;
	updateStrand <<< blocks_per_grid, g_threadsPerBlock >>> (m_numVertices, m_stepper->getVtxToStrand(), strand_vtx, m_stepper->getX());
	updateQuad <<< blocks_per_grid, g_threadsPerBlock >>> (
		m_numVertices, slides, m_stepper->getVtxToStrand(), m_stepper->getStrandPtr(), strand_vtx, 
		m_stepper->getTangents(), m_stepper->getMaterialFrames1(), m_radius, quad_vtx);

	// Upmap data
	cudaGraphicsUnmapResources(1, &m_cudaStrandVerticesResource, 0);
	cudaGraphicsUnmapResources(1, &m_cudaQuadVerticesResource, 0);
}
