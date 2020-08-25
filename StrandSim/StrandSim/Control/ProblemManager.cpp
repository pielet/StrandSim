#include "ProblemManager.h"
#include <fstream>
#include "../Dynamic/ElasticStrand.h"
#include "../Dynamic/StepperManager.h"
#include "../Render/StrandRenderer.h"

ProblemManager::ProblemManager(const std::string& xml_file)
{
	
	loadXMLFile(xml_file);
}

ProblemManager::~ProblemManager()
{
	clear();
}

void ProblemManager::loadXMLFile(const std::string& xml_file)
{
	// Check file path
	std::ifstream textfile(xml_file.c_str(), std::ifstream::in);
	if (!textfile)
	{
		std::cerr << "Can NOT open xml file: " << xml_file << ". Exiting." << std::endl;
		exit(-1);
	}
	// Read content
	std::vector<char> content;
	std::string line;
	while (std::getline(textfile, line))
	{
		for (int i = 0; i < line.size(); ++i) content.push_back(line[i]);
	}
	content.push_back('\0');
	textfile.close();

	rapidxml::xml_document<> doc;
	doc.parse<0>(content.data());
	rapidxml::xml_node<>* scene = doc.first_node("scene");

	// Load parameters
	loadParameters(scene);
	std::cout << "Finish loading parameters." << std::endl;

	// Load hairs
	int num_strands = 0;
	for (rapidxml::xml_node<>* nd = scene->first_node("Strand"); nd; nd = nd->next_sibling("Strand")) ++num_strands;
	m_strands.reserve(num_strands);
	m_streams.resize(num_strands);
	int idx = 0;
	for (rapidxml::xml_node<>* nd = scene->first_node("Strand"); nd; nd = nd->next_sibling("Strand"))
	{
		cudaStreamCreate(&m_streams[idx]);
		loadStrand(idx, nd, m_streams[idx]);
		++idx;
	}
	std::cout << "Finish loading " << num_strands << " strands." << std::endl;

	// Create stepper
	m_stepper = new StepperManager(m_strands, &m_simulationParams, m_dt);
	std::cout << "Created stepper manager." << std::endl;
}

void ProblemManager::clear()
{
	delete m_stepper;
	for (int i = 0; i < m_strands.size();++i) 
	{
		delete m_renderer[i];
		delete m_strands[i];
		cudaStreamDestroy(m_streams[i]);
	}
}

void ProblemManager::step()
{
	// Execute scripts

	m_stepper->step();

	cudaDeviceSynchronize();

	for (StrandRenderer* renderer : m_renderer)
		renderer->ackGeometryChange();
}

void ProblemManager::render(const Shader* shader)
{
	for (StrandRenderer* renderer : m_renderer)
		renderer->render(shader);
}

void ProblemManager::loadParameters(rapidxml::xml_node<>* doc)
{
	loadParam(doc, "duration", m_duration);
	loadParam(doc, "dt", m_dt);

	rapidxml::xml_node<>* nd = doc->first_node("SimulationParameters");
	loadParam(nd, "maxNewtonIterations", m_simulationParams.m_maxNewtonIterations);
	loadParam(nd, "innerIterations", m_simulationParams.m_innerIterations);
	loadParam(nd, "postProcessIterations", m_simulationParams.m_postProcessIterations);
	loadParam(nd, "newtonTolerance", m_simulationParams.m_newtonTolerance);
	loadParam(nd, "collisionTolerance", m_simulationParams.m_collisionTolerance);
	loadParam(nd, "warmStarting", m_simulationParams.m_warmStarting);
	loadParam(nd, "relaxationFactor", m_simulationParams.m_relaxationFactor);
	loadParam(nd, "useQuasiNewton", m_simulationParams.m_useQuasiNewton);
	loadParam(nd, "windowSize", m_simulationParams.m_windowSize);
	loadParam(nd, "useLineSearch", m_simulationParams.m_useLineSearch);
	loadParam(nd, "lsAlpha", m_simulationParams.m_lsAlpha);
	loadParam(nd, "lsBeta", m_simulationParams.m_lsBeta);
	loadParam(nd, "solveCollision", m_simulationParams.m_solveCollision);
	loadParam(nd, "pruneExternalCollision", m_simulationParams.m_pruneExternalCollision);
	loadParam(nd, "pruneSelfCollision", m_simulationParams.m_pruneSelfCollision);

	for (rapidxml::xml_node<>* nd = doc->first_node("StrandParameters"); nd; nd = nd->next_sibling("StrandParameters"))
	{
		StrandParameters param;
		loadParam(nd, "radius", param.m_radius);
		loadParam(nd, "youngsModulus", param.m_youngsModulus);
		loadParam(nd, "poissonRatio", param.m_poissonRatio);
		loadParam(nd, "density", param.m_density);
		loadParam(nd, "stretchMultiplier", param.m_stretchMultiplier);
		param.m_shearModulus = param.m_youngsModulus / (1 + param.m_poissonRatio) / 2;
		param.m_ks = M_PI * square(param.m_radius) * param.m_youngsModulus;
		param.m_kt = M_PI / 2 * biquad(param.m_radius) * param.m_shearModulus;
		param.m_kb = M_PI / 4 * biquad(param.m_radius) * param.m_youngsModulus;
		m_strandParams.push_back(param);
	}

	nd = doc->first_node("CollisionParameters");
	loadParam(nd, "hairHairFrictionCoefficient", m_collisionParams.m_hairHairFrictionCoefficient);
	loadParam(nd, "hairMeshFrictionCoefficient", m_collisionParams.m_hairMeshFrictionCoefficient);
	loadParam(nd, "selfCollisionRadius", m_collisionParams.m_selfCollisionRadius);
	loadParam(nd, "externalCollisionRadius", m_collisionParams.m_externalCollisionRadius);
	loadParam(nd, "repulseRadius", m_collisionParams.m_repulseRadius);
}

void ProblemManager::loadStrand(int global_idx, rapidxml::xml_node<>* nd, cudaStream_t stream)
{
	int num_particles = 1;
	loadAttrib(nd, "count", num_particles);
	Eigen::VecXx particles = Eigen::VecXx::Zero(4 * num_particles - 1);

	int param = 0;
	loadAttrib(nd, "param", param);
	if (param >= m_strandParams.size())
	{
		std::cerr << "Strand " << global_idx << " has WRONG parameter index" << std::endl;
		exit(-1);
	}

	FixedMap fixed_map;

	int idx = 0;
	for (rapidxml::xml_node<>* subnd = nd->first_node("particle"); subnd; subnd = subnd->next_sibling("particle"))
	{
		Eigen::Vec3x pos;
		int fixed = 0;

		std::string pos_string = subnd->first_attribute("x")->value();
		std::stringstream(pos_string) >> pos(0) >> pos(1) >> pos(2);
		particles.segment<3>(4 * idx) = pos;

		loadAttrib(subnd, "fixed", fixed);
		if (fixed > 0)
		{
			fixed_map[idx] = pos;
		}
		++idx;
	}
	assert(idx == num_particles);

	m_strands.push_back(new ElasticStrand(global_idx, particles.size(), particles.data(), fixed_map, &m_strandParams[param], stream));
	m_renderer.push_back(new StrandRenderer(m_strands.back(), stream));
}

template<typename T>
static bool ProblemManager::loadParam(rapidxml::xml_node<>* nd, const char* name, T& param)
{
	rapidxml::xml_node<>* subnd = nd->first_node(name);
	if (subnd)
	{
		std::string attrib(subnd->first_attribute("value")->value());
		if (!bool(std::stringstream(attrib) >> param))
		{
			std::cerr << "Failed to parse value of " << name << " attribute for " << nd->name() << ". Exiting." << std::endl;
			exit(-1);
		}
		return true;
	}
	else return false;
}

template<typename T>
static bool ProblemManager::loadAttrib(rapidxml::xml_node<>* nd, const char* name, T& param)
{
	rapidxml::xml_attribute<>* attrib = nd->first_attribute(name);
	if (attrib)
	{
		if (!bool(std::stringstream(attrib->value()) >> param))
		{
			std::cerr << "Failed to parse value of " << name << " attribute for " << nd->name() << ". Exiting." << std::endl;
			exit(-1);
		}
		return true;
	}
	else return false;
}
