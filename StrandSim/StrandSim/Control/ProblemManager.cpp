#include "ProblemManager.h"
#include <fstream>
#include "../Dynamic/StepperManager.h"
#include "../Render/StrandRenderer.h"

int g_threadsPerBlock = 32;
int g_maxVertex = 1;

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
	loadHairs(scene);
	for (auto nd = scene->first_node("hairobj"); nd; nd = nd->next_sibling("hairobj"))
		loadHairobj(nd);
	int num_strands = m_strand_ptr.size() - 1;
	int num_particles = m_particle_x.size();
	std::cout << "Finish loading " << num_particles << " particles and " << num_strands << " strands." << std::endl;

	// Wrap data into device-friendly array
	int num_dofs = 4 * num_particles - num_strands;
	Eigen::VecXx v(num_dofs);
	Eigen::VecXx x(num_dofs);
	v.setZero();
	x.setZero();

	for (int i = 0; i < num_strands; ++i)
	{
		g_maxVertex = std::max(g_maxVertex, m_strand_ptr[i + 1] - m_strand_ptr[i]);
		for (int j = m_strand_ptr[i]; j < m_strand_ptr[i + 1]; ++j)
		{
			x.segment<3>(4 * j - i) = m_particle_x[j];
			v.segment<3>(4 * j - i) = m_particle_v[j];
		}
	}

	int num_fixed = m_fixed.size();
	std::vector<int> fixed_idx(num_fixed);
	Eigen::VecXx fixed_pos(3 * num_fixed);
	int idx = 0;
	for (const auto& item : m_fixed)
	{
		fixed_idx[idx] = item.first;
		fixed_pos.segment<3>(3 * idx) = item.second;
		++idx;
	}

	m_stepper = new StepperManager(
		m_dt, num_dofs, num_fixed, x.data(), v.data(), m_strand_ptr, fixed_idx.data(), fixed_pos.data(), 
		&m_simulationParams, &m_strandParams, &m_collisionParams);
	m_renderer = new StrandRenderer(num_particles, num_particles - num_strands, m_strandParams.m_radius, m_strand_ptr, m_stepper);

	std::cout << "Finish initialization." << std::endl;
}

void ProblemManager::clear()
{
	delete m_stepper;
	delete m_renderer;
}

void ProblemManager::step()
{
	// Execute scripts

	m_stepper->step();

	cudaDeviceSynchronize();

	m_renderer->ackGeometryChange();
}

void ProblemManager::render(const Shader* shader)
{
	m_renderer->render(shader);
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

	nd = doc->first_node("StrandParameters");
	loadParam(nd, "radius", m_strandParams.m_radius);
	loadParam(nd, "youngsModulus", m_strandParams.m_youngsModulus);
	loadParam(nd, "poissonRatio", m_strandParams.m_poissonRatio);
	loadParam(nd, "density", m_strandParams.m_density);
	loadParam(nd, "stretchMultiplier", m_strandParams.m_stretchMultiplier);
	m_strandParams.m_shearModulus = m_strandParams.m_youngsModulus / ((1 + m_strandParams.m_poissonRatio) * 2);
	m_strandParams.m_ks = M_PI * square(m_strandParams.m_radius) * m_strandParams.m_youngsModulus * m_strandParams.m_stretchMultiplier;
	m_strandParams.m_kt = M_PI / 2 * biquad(m_strandParams.m_radius) * m_strandParams.m_shearModulus;
	m_strandParams.m_kb = M_PI / 4 * biquad(m_strandParams.m_radius) * m_strandParams.m_youngsModulus;

	nd = doc->first_node("CollisionParameters");
	loadParam(nd, "hairHairFrictionCoefficient", m_collisionParams.m_hairHairFrictionCoefficient);
	loadParam(nd, "hairMeshFrictionCoefficient", m_collisionParams.m_hairMeshFrictionCoefficient);
	loadParam(nd, "selfCollisionRadius", m_collisionParams.m_selfCollisionRadius);
	loadParam(nd, "externalCollisionRadius", m_collisionParams.m_externalCollisionRadius);
	loadParam(nd, "repulseRadius", m_collisionParams.m_repulseRadius);
}

void ProblemManager::loadHairs(rapidxml::xml_node<>* doc)
{
	int num_particles = 0;
	int num_strands = 0;
	for (rapidxml::xml_node<>* nd = doc->first_node("particle"); nd; nd = nd->next_sibling("particle")) ++num_particles;
	for (rapidxml::xml_node<>* nd = doc->first_node("hair"); nd; nd = nd->next_sibling("hair")) ++num_strands;
	m_particle_x.resize(num_particles);
	m_particle_v.resize(num_particles);
	m_strand_ptr.resize(num_strands + 1);

	// load particles
	int idx = 0;
	Eigen::Vec3x pos, vel;
	int fixed;
	std::string atrrib;
	for (rapidxml::xml_node<>* nd = doc->first_node("particle"); nd; nd = nd->next_sibling("particle"))
	{
		atrrib = nd->first_attribute("x")->value();
		std::stringstream(atrrib) >> pos(0) >> pos(1) >> pos(2);
		atrrib = nd->first_attribute("v")->value();
		std::stringstream(atrrib) >> vel(0) >> vel(1) >> vel(2);

		fixed = 0;
		loadAttrib(nd, "fixed", fixed);
		if (fixed) m_fixed[idx] = pos;

		m_particle_x[idx] = pos;
		m_particle_v[idx] = vel;

		idx++;
	}

	// load strands
	int start;
	idx = 0;
	for (rapidxml::xml_node<>* nd = doc->first_node("hair"); nd; nd = nd->next_sibling("hair"))
	{
		loadAttrib(nd, "start", start);
		m_strand_ptr[idx++] = start;
	}
	m_strand_ptr[idx] = num_particles;
}

void ProblemManager::loadHairobj(rapidxml::xml_node<>* nd)
{
	std::string hairobj_file;
	loadAttrib(nd, "filename", hairobj_file);
	if (hairobj_file.empty())
	{
		std::cerr << "Must input a hairobj file. EXIT." << std::endl;
		exit(-1);
	}

	std::string line;
	std::ifstream ifs(hairobj_file);

	if (ifs.fail()) {
		std::cerr << "Failed to read file: " << hairobj_file << ". EXIT." << std::endl;
		exit(-1);
	}

	// load fixed indices info
	auto fixed_node = nd->first_node("fixed");
	int start_vtx = 0;
	int end_vtx = 0;
	if (fixed_node)
	{
		loadAttrib(fixed_node, "start", start_vtx);
		loadAttrib(fixed_node, "end", end_vtx);
	}

	// check particles and strands number
	int num_particles = 0;
	int num_strands = 0;
	while (std::getline(ifs, line))
	{
		if (line[0] == 'v') ++num_particles;
		if (line[0] == 'l') ++num_strands;
	}
	int vtx_base = m_particle_x.size();
	int strand_base = m_strand_ptr.size() - 1;
	m_particle_x.resize(vtx_base + num_particles);
	m_particle_v.resize(vtx_base + num_particles, Eigen::Vec3x::Zero());
	m_strand_ptr.resize(strand_base + num_strands + 1);

	// load
	int vtx_idx = 0;
	int strand_idx = 0;
	ifs.clear();
	ifs.seekg(0);
	while (std::getline(ifs, line))
	{
		std::vector<std::string> tokens = tokenize(line, ' ');
		if (tokens[0] == "v" && tokens.size() == 4)		// vertex
		{
			Eigen::Vec3x pos;
			for (int i = 0; i < 3; ++i) {
				std::stringstream(tokens[i + 1]) >> pos(i);
			}
			m_particle_x[vtx_base + vtx_idx] = pos;
			++vtx_idx;
		}
		else if (tokens[0] == "l" && tokens.size() > 1)		// hair
		{
			int idx;
			std::stringstream(tokens[1]) >> idx;
			m_strand_ptr[strand_base + strand_idx] = vtx_base + idx - 1;
			++strand_idx;

			// fixed points
			for (int i = start_vtx; i < end_vtx; ++i)
			{
				std::stringstream(tokens[i + 1]) >> idx;
				m_fixed[vtx_base + idx - 1] = m_particle_x[vtx_base + idx - 1];
			}
		}
	}
	assert(vtx_idx == num_particles);
	assert(strand_idx == num_strands);
	m_strand_ptr[strand_base + strand_idx] = vtx_base + vtx_idx;

	ifs.close();
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

std::vector<std::string> ProblemManager::tokenize(const std::string& str, const char chr)
{
	std::vector<std::string> tokens;

	std::string::size_type substring_start = 0;
	std::string::size_type substring_end = str.find_first_of(chr, substring_start);
	while (substring_end != std::string::npos)
	{
		tokens.emplace_back(str.substr(substring_start, substring_end - substring_start));
		substring_start = substring_end + 1;
		substring_end = str.find_first_of(chr, substring_start);
	}
	// Grab the trailing substring, if present
	if (substring_start < str.size())
	{
		tokens.emplace_back(str.substr(substring_start));
	}
	// Case of final character the delimiter
	if (str.back() == chr)
	{
		tokens.emplace_back("");
	}

	return tokens;
}