#include "StepperManager.h"
#include "ElasticStrand.h"
#include "NewtonStepper.h"

StepperManager::StepperManager(const std::vector<ElasticStrand*>& strands, const SimulationParameters* params, Scalar dt) :
	m_time(0.),
	m_dt(dt),
	m_numStrands(strands.size()),
	m_params(params),
	m_strands(strands)
{
	m_steppers.reserve(m_numStrands);
	for (int i = 0; i < m_numStrands; ++i)
	{
		m_steppers.push_back(new NewtonStepper(i, strands[i], params, m_strands[i]->getStream()));
	}
}

StepperManager::~StepperManager()
{
	for (int i = 0; i < m_numStrands; ++i)
	{
		delete m_steppers[i];
	}
}

void StepperManager::step()
{
	std::cout << "[Prepare Simulation Step]" << std::endl;
#pragma omp parallel for
	for (int i = 0; i < m_numStrands; ++i)
	{
		m_steppers[i]->prepareStep(m_dt);
	}

	std::cout << "[Step Dynamics]" << std::endl;
	std::vector<bool> pass(m_numStrands, false);
	int k = 0;
	for (; k < m_params->m_maxNewtonIterations; ++k)
	{
#pragma omp parallel for
		for (int i = 0; i < m_numStrands; ++i)
		{
			pass[i] = m_steppers[i]->performOneIteration();
		}

		bool all_pass = true;
		for (bool p : pass) all_pass = all_pass && p;
		if (all_pass) break;
	}

	std::cout << "[Post Dynamics]" << std::endl;
#pragma omp parallel for
	for (int i = 0; i < m_numStrands; ++i)
	{
		m_steppers[i]->commitVelocities();
	}
}