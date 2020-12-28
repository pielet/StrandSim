#ifndef PROBLEM_MANAGER_H
#define PROBLEM_MANAGER_H

#include <iostream>
#include <vector>
#include <map>
#include <rapidxml.hpp>
#include <cuda_runtime.h>

#include "Parameters.h"
#include "../Utils/EigenDef.h"

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
	void loadHairs(rapidxml::xml_node<>* doc);
	void loadHairobj(rapidxml::xml_node<>* doc);

	void loadExternalObject();
	void loadScript();

	template<typename T>
	static bool loadParam(rapidxml::xml_node<>* nd, const char* name, T& param);

	template<typename T>
	static bool loadAttrib(rapidxml::xml_node<>* nd, const char* name, T& param);

	static std::vector<std::string> tokenize(const std::string& str, const char chr);

	Scalar m_duration;
	Scalar m_dt;

	SimulationParameters m_simulationParams;
	StrandParameters m_strandParams;
	CollisionParameters m_collisionParams;

	std::vector<Eigen::Vec3x> m_particle_x;
	std::vector<Eigen::Vec3x> m_particle_v;

	std::map<int, Eigen::Vec3x> m_fixed;

	std::vector<int> m_strand_ptr;

	StrandRenderer* m_renderer;
	StepperManager* m_stepper;
};


#endif // !PROBLEM_MANAGER_H
