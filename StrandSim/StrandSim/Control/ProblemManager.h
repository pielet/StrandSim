#ifndef PROBLEM_MANAGER_H
#define PROBLEM_MANAGER_H

#include <iostream>
#include <vector>
#include <rapidxml.hpp>
#include <cuda_runtime.h>

#include "Parameters.h"
#include "../Utils/EigenDef.h"

class ElasticStrand;
class StrandRenderer;
class StepperManager;
class Shader;

class ProblemManager
{
public:
	ProblemManager(const std::string& xml_file);
	~ProblemManager();

	void loadXMLFile(const std::string& xml_file);
	void clear();

	void step();
	void render(const Shader* shader);

	void saveModel();
	void saveCheckpoint();
	void loadCheckpoint();

protected:
	void loadParameters(rapidxml::xml_node<>* doc);

	void loadStrand(int global_idx, rapidxml::xml_node<>* nd, cudaStream_t stream);

	void loadExternalObject();
	void loadScript();

	template<typename T>
	static bool loadParam(rapidxml::xml_node<>* nd, const char* name, T& param);

	template<typename T>
	static bool loadAttrib(rapidxml::xml_node<>* nd, const char* name, T& param);

	Scalar m_duration;
	Scalar m_dt;

	SimulationParameters m_simulationParams;
	std::vector<StrandParameters> m_strandParams;
	CollisionParameters m_collisionParams;

	std::vector<ElasticStrand*> m_strands;
	std::vector<StrandRenderer*> m_renderer;
	std::vector<cudaStream_t> m_streams;
	StepperManager* m_stepper;
};


#endif // !PROBLEM_MANAGER_H
