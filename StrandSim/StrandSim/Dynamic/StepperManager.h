#ifndef STEPPER_MANAGER_H
#define STEPPER_MANAGER_H

#include <iostream>
#include <vector>
#include "../Control/Parameters.h"

class ElasticStrand;
class NewtonStepper;

class StepperManager
{
public:
	StepperManager(const std::vector<ElasticStrand*>& strands, const SimulationParameters* params, Scalar dt);
	~StepperManager();

	void step();

protected:
	Scalar m_time;  //< Current time
	Scalar m_dt;    //< Timer per frame
	int m_numStrands;

	const SimulationParameters* m_params;
	const std::vector<ElasticStrand*> m_strands;
	std::vector<NewtonStepper*> m_steppers;
};

#endif // !STEPPER_MANAGER_H
