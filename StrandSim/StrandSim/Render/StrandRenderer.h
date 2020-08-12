#ifndef STRAND_RENDERER_H
#define STRAND_RENDERER_H

#include <GL/glew.h>
#include <cuda_runtime.h>

class ElasticStrand;
class Shader;

class StrandRenderer
{
public:
	StrandRenderer(const ElasticStrand* strand, cudaStream_t stream);
	~StrandRenderer();

	void render(const Shader* shader);
	void ackGeometryChange() { m_geometryChanged = true; }

protected:
	void updateVertices();

	cudaStream_t m_stream;
	const ElasticStrand* m_strand;

	bool m_geometryChanged;

	GLuint m_strandVerticesBuffer;
	GLuint m_quadVerticesBuffer;
	GLuint m_quadIndicesBuffer;

	struct cudaGraphicsResource* m_cudaStrandVerticesResource;
	struct cudaGraphicsResource* m_cudaQuadVerticesResource;
};

#endif // !STRAND_RENDERER_H
