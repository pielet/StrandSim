#ifndef PARAMETERS_H
#define PARAMETERS_H

typedef double Scalar;
const Scalar M_PI = 3.14159265358979323846;

struct SimulationParameters
{
	/* Iteration */
	int m_maxNewtonIterations = 10;
	int m_innerIterations = 1;
	int m_postProcessIterations = 0;
	Scalar m_newtonTolerance = 1e-6;
	Scalar m_collisionTolerance = 1e-6;

	/* Warm-starting */
	bool m_warmStarting = false;
	Scalar m_relaxationFactor = 0.5;

	/* Quasi-Newton */
	bool m_useQuasiNewton = false;
	int m_windowSize = 0;

	/* Line search */
	bool m_useLineSearch = false;
	Scalar m_lsAlpha = 0.03;
	Scalar m_lsBeta = 0.5;

	/* Collision */
	bool m_solveCollision = true;
	bool m_pruneExternalCollision = false;
	bool m_pruneSelfCollision = false;

	/* Debug */
};

struct StrandParameters
{
	/* Input */
	Scalar m_radius = 0.005;
	Scalar m_youngsModulus = 1e10;
	Scalar m_poissonRatio = 0.36;
	Scalar m_density = 1.32;
	Scalar m_stretchMultiplier = 1e-2;

	/* Derivative */
	Scalar m_shearModulus;
	Scalar m_ks;
	Scalar m_kt;
	Scalar m_kb;
};

struct CollisionParameters
{
	Scalar m_hairHairFrictionCoefficient = 0.3;
	Scalar m_hairMeshFrictionCoefficient = 0.3;
	Scalar m_selfCollisionRadius = 0.0055;
	Scalar m_externalCollisionRadius = 0.0055;
	Scalar m_repulseRadius = 0.005;
};

#endif // !PARAMETERS_H