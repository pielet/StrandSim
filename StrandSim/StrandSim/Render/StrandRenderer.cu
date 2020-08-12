#include "StrandRenderer.h"

// CUDA
#include <cuda_gl_interop.h>
#include <vector>
#include "../Utils/CUDAMathDef.h"
#include "../Render/Shader.h"
#include "../Dynamic/ElasticStrand.h"

const int slides = 8;

__global__ void updateStrand(Vec3f* strand_vtx, const Scalar* x)
{
	int i = threadIdx.x;

	strand_vtx[i] = Vec3f(x[4 * i], x[4 * i + 1], x[4 * i + 2]);
}

__global__ void updateQuad(int slides, Vec3f* quad_vtx, const Vec3f* vtx, const Vec3x* tangents, const Vec3x* materialFrame1, float radius)
{
	int i = threadIdx.x;

	Vec3x tangent;
	Vec3x normal;
	if (i < blockDim.x - 1)
	{
		tangent = tangents[i];
		normal = materialFrame1[i];
	}
	else
	{
		tangent = tangents[i - 1];
		normal = materialFrame1[i - 1];
	}

	for (int j = 0; j < slides; ++j)
	{
		quad_vtx[slides * i + j] = vtx[i] + radius * Vec3f(normal(0), normal(1), normal(2));
		rotateAxisAngle(normal, tangent, 2. / slides);
	}
}

StrandRenderer::StrandRenderer(const ElasticStrand* strand, cudaStream_t stream) :
	m_strand(strand),
	m_geometryChanged(true),
	m_stream(stream)
{
	int num_vertices = strand->getNumVertices();

	// Init buffers
	glGenBuffers(1, &m_strandVerticesBuffer);
	glGenBuffers(1, &m_quadVerticesBuffer);
	glGenBuffers(1, &m_quadIndicesBuffer);

	// Generate vertice buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_strandVerticesBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m_quadVerticesBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * slides * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);

	// Buffer indices data
	std::vector<unsigned> indice((num_vertices - 1) * slides * 4);
	int i = 0;
	int cur_idx = 0;
	for (int j = 0; j < num_vertices - 1; ++j)
	{
		for (int k = 0; k < slides; ++k) 
		{
			int k1 = (k + 1) % slides;
			indice[i++] = cur_idx + k;	
			indice[i++] = cur_idx + k1;
			indice[i++] = cur_idx + k1 + slides;
			indice[i++] = cur_idx + k + slides;
		}
		cur_idx += slides;
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_quadIndicesBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indice.size() * sizeof(unsigned), indice.data(), GL_STATIC_DRAW);

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

	int num_vertices = m_strand->getNumVertices();

	glEnableVertexAttribArray(0);

	// Draw strand line
	glBindBuffer(GL_ARRAY_BUFFER, m_strandVerticesBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	shader->setVec4f("vertex_color", 0.f, 0.f, 0.f, 1.f);  // black
	glDrawArrays(GL_LINE_STRIP, 0, num_vertices);

	// Draw quads
	glBindBuffer(GL_ARRAY_BUFFER, m_quadVerticesBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_quadIndicesBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	shader->setVec4f("vertex_color", 0.f, 0.f, 1.f, 0.2f);  // transparent blue
	glDrawElements(GL_QUADS, (num_vertices - 1) * slides * 4, GL_UNSIGNED_INT, 0);
}

void StrandRenderer::updateVertices()
{
	// Map data
	Vec3f* strand_vtx;
	Vec3f* quad_vtx;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &m_cudaStrandVerticesResource, m_stream);
	cudaGraphicsResourceGetMappedPointer((void**)& strand_vtx, &num_bytes, m_cudaStrandVerticesResource);
	cudaGraphicsMapResources(1, &m_cudaQuadVerticesResource, m_stream);
	cudaGraphicsResourceGetMappedPointer((void**)& quad_vtx, &num_bytes, m_cudaQuadVerticesResource);

	// Launch kernal
	int num_vertices = m_strand->getNumVertices();
	updateStrand <<< 1, num_vertices, 0, m_stream >>> (strand_vtx, m_strand->getX());
	updateQuad <<< 1, num_vertices, 0, m_stream >>> (slides, quad_vtx, strand_vtx, m_strand->getCurrentState()->m_tangents, 
		m_strand->getCurrentState()->m_materialFrames1, m_strand->getParameters()->m_radius);

	// Upmap data
	cudaGraphicsUnmapResources(1, &m_cudaStrandVerticesResource, m_stream);
	cudaGraphicsUnmapResources(1, &m_cudaQuadVerticesResource, m_stream);
}
