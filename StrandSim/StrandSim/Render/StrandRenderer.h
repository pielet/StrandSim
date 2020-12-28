#ifndef STRAND_RENDERER_H
#define STRAND_RENDERER_H

#include <vector>
#include <GL/glew.h>
#include <cuda_runtime.h>

class Shader;
class StepperManager;

class StrandRenderer
{
public:
	StrandRenderer(int num_vertices, int num_edges, float radius, const std::vector<int>& strand_ptr, const StepperManager* stepper);
	~StrandRenderer();

	void render(const Shader* shader);
	void ackGeometryChange() { m_geometryChanged = true; }

protected:
	void updateVertices();

	int m_numVertices;
	int m_numEdges;
	float m_radius;
	const std::vector<int>& m_strand_ptr;

	const StepperManager* m_stepper;

	bool m_geometryChanged;

	GLuint m_strandVerticesBuffer;
	GLuint m_strandIndicesBuffer;
	GLuint m_quadVerticesBuffer;
	GLuint m_quadIndicesBuffer;

	struct cudaGraphicsResource* m_cudaStrandVerticesResource;
	struct cudaGraphicsResource* m_cudaQuadVerticesResource;
};

#endif // !STRAND_RENDERER_H
