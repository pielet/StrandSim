#include "BandMatrix.h"
#include "StrandForces.h"
#include "../Utils/CUDAMathDef.h"
#include "ElasticStrand.h"
#include "StrandState.h"
#include <iostream>

__global__ void computeFixingEnergy(Scalar ks, Scalar* fixingEnergy, const int* indices, const Scalar* x, const Vec3x* positions)
{
	int i = threadIdx.x;
	int j = indices[i];

	extern __shared__ Scalar energy[];
	Vec3x cur_pos(x[4 * j], x[4 * j + 1], x[4 * j + 2]);
	energy[i] = 0.5 * ks * (positions[i] - cur_pos).norm2();
	__syncthreads();

	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
		{
			energy[i] += energy[i + strip];
		}
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	fixingEnergy[i] = energy[i];
}

__global__ void computeFixingForce(Scalar ks, Scalar* forces, const int* indices, const Scalar* x, const Vec3x* positions)
{
	int i = threadIdx.x;
	int j = indices[i];

	forces[4 * j] += ks * (positions[i](0) - x[4 * j]);
	forces[4 * j + 1] += ks * (positions[i](1) - x[4 * j + 1]);
	forces[4 * j + 2] += ks * (positions[i](2) - x[4 * j + 2]);
}

__global__ void computeFixingGradient(Scalar ks, MatrixWrapper<Scalar>* gradient, const int* indices)
{
	int i = threadIdx.x;
	int j = indices[i];

	gradient->addInPlace(4 * j, 4 * j, ks);
	gradient->addInPlace(4 * j + 1, 4 * j + 1, ks);
	gradient->addInPlace(4 * j + 2, 4 * j + 2, ks);
}

__global__ void computeStretchingEnergy(Scalar ks, Scalar* stretchingEnergy, const Scalar* lengths, const Scalar* restLengths)
{
	int i = threadIdx.x;

	extern __shared__ Scalar energy[];
	energy[i] = 0.5 * ks * (lengths[i] - restLengths[i]) * (lengths[i] - restLengths[i]) / restLengths[i];
	__syncthreads();

	// Reduction
	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
		{
			energy[i] += energy[i + strip];
		}
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	stretchingEnergy[i] = energy[i];
}

__global__ void computeStretchingForces(Scalar ks, Scalar* forces, const Scalar* lengths, const Scalar* restLengths, const Vec3x* tangents)
{
	int i = threadIdx.x;

	Vec3x f = ks * (lengths[i] / restLengths[i] - 1) * tangents[i];
	
	forces[4 * i] += f(0);
	forces[4 * i + 1] += f(1);
	forces[4 * i + 2] += f(2);
	__syncthreads();
	forces[4 * i + 4] -= f(0);
	forces[4 * i + 5] -= f(1);
	forces[4 * i + 6] -= f(2);
}

__global__ void computeStretchingGradient(Scalar ks, MatrixWrapper<Scalar>* gradient, const Scalar* lengths, 
	const Scalar* restLengths, const Vec3x* tangents)
{
	int i = threadIdx.x;

	Mat3x M = ks * (1.0 / lengths[i] * outerProd(tangents[i]));
	if (lengths[i] > restLengths[i])
		M += ks * (1.0 / restLengths[i] - 1.0 / lengths[i]) * Mat3x::Identity();

	// Accumulte to hessian matrix
	int j, k;
	for (j = 0; j < 3; ++j)
	{
		for (k = 0; k < 3; ++k)
		{
			gradient->addInPlace(4 * i + j, 4 * i + k, M(j, k));
			gradient->addInPlace(4 * i + j, 4 * i + 4 + k, -M(j, k));
			gradient->addInPlace(4 * i + 4 + j, 4 * i + k, -M(j, k));
		}
	}
	__syncthreads();
	for (j = 0; j < 3; ++j)
	{
		for (k = 0; k < 3; ++k)
		{
			gradient->addInPlace(4 * i + 4 + j, 4 * i + 4 + k, M(j, k));
		}
	}
}

__global__ void computeGradTwist(Vec11x* gradTwists, const Vec3x* curvatureBinormals, const Scalar* lengths)
{
	int i = threadIdx.x;

	Vec11x& Dtwist = gradTwists[i];
	const Vec3x& kb = curvatureBinormals[i];

	Dtwist.assignAt(0, -0.5 / lengths[i] * kb);
	Dtwist.assignAt(8, 0.5 / lengths[i + 1] * kb);
	Dtwist.assignAt(4, -(Dtwist.at<3>(0) + Dtwist.at<3>(8)));
	Dtwist(3) = -1;
	Dtwist(7) = 1;
}

__global__ void computeHessTwist(Mat11x* hessTwists, const Vec3x* tangents, const Scalar* lengths, const Vec3x* curvatureBinormals)
{
	int vtx = threadIdx.x;

	Mat11x& DDtwist = hessTwists[vtx];
	DDtwist.setZero();

	const Vec3x& te = tangents[vtx];
	const Vec3x& tf = tangents[vtx + 1];
	const Scalar norm_e = lengths[vtx];
	const Scalar norm_f = lengths[vtx + 1];
	const Vec3x& kb = curvatureBinormals[vtx];

	Scalar chi = MAX(1e-12, 1 + te.dot(tf));

	const Vec3x tilde_t = 1.0 / chi * (te + tf);

	const Mat3x D2mDe2 = -0.25 / (norm_e * norm_e) * (outerProd(kb, te + tilde_t) + outerProd(te + tilde_t, kb));
	const Mat3x D2mDf2 = -0.25 / (norm_f * norm_f) * (outerProd(kb, tf + tilde_t) + outerProd(tf + tilde_t, kb));
	const Mat3x D2mDeDf = 0.5 / (norm_e * norm_f) * (2.0 / chi * crossMat(te) - outerProd(kb, tilde_t));
	const Mat3x D2mDfDe = D2mDeDf.transpose();

	DDtwist.assignAt(0, 0, D2mDe2);
	DDtwist.assignAt(0, 4, -D2mDe2 + D2mDeDf);
	DDtwist.assignAt(4, 0, -D2mDe2 + D2mDfDe);
	DDtwist.assignAt(4, 4, D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2);
	DDtwist.assignAt(0, 8, -D2mDeDf);
	DDtwist.assignAt(8, 0, -D2mDfDe);
	DDtwist.assignAt(8, 4, D2mDfDe - D2mDf2);
	DDtwist.assignAt(4, 8, D2mDeDf - D2mDf2);
	DDtwist.assignAt(8, 8, D2mDf2);

	assert(isSymmetric(DDtwist));
}

__global__ void computeTwistingEnergy(Scalar kt, Scalar* twistingEnergy, const Scalar* twists, const Scalar* restTwists, const Scalar* ilens)
{
	int i = threadIdx.x;

	extern __shared__ Scalar energy[];
	energy[i] = 0.5 * kt * (twists[i] - restTwists[i]) * (twists[i] - restTwists[i]) * ilens[i + 1];
	__syncthreads();

	// Reduction
	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
		{
			energy[i] += energy[i + strip];
		}
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	twistingEnergy[i] = energy[i];
}

__global__ void computeTwistingForce(Scalar kt, Scalar* totalForces, const Scalar* twists, const Scalar* restTwists, 
	const Scalar* ilens, const Vec11x* gradTwists)
{
	int i = threadIdx.x;

	Vec11x force = -kt * ilens[i + 1] * (twists[i] - restTwists[i]) * gradTwists[i];

	// Add to total forces
	int j;
	for (j = 0; j < 4; ++j) 
		totalForces[4 * i + j] += force(j);
	__syncthreads();
	for (j = 4; j < 8; ++j)
		totalForces[4 * i + j] += force(j);
	__syncthreads();
	for (j = 8; j < 11; ++j)
		totalForces[4 * i + j] += force(j);
}

__global__ void computeTwistingGradient(Scalar kt, MatrixWrapper<Scalar>* gradient, const Scalar* twists, const Scalar* restTwists, 
	const Scalar* ilens, const Vec11x* gradTwists, const Mat11x* hessTwists)
{
	int i = threadIdx.x;

	Mat11x localJ = kt * ilens[i + 1] * (outerProd(gradTwists[i]) + (twists[i] - restTwists[i]) * hessTwists[i]);

	// Add to total hessian
	int j, k;
	for (j = 0; j < 4; ++j) // J11, J12, J13
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
	__syncthreads();
	for (j = 4; j < 8; ++j) // J21, J22, J23
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
	__syncthreads();
	for (j = 8; j < 11; ++j) // J31, J32, J33
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
}

__global__ void computeGradKappa(Mat11x4x* gradKappas, const Scalar* lengths, const Vec3x* tangents, const Vec3x* curvatureBinormals, 
	const Vec3x* materialFrames1, const Vec3x* materialFrames2, const Vec4x* kappas)
{
	int vtx = threadIdx.x;

	Mat11x4x& gradKappa = gradKappas[vtx];

	const Scalar norm_e = lengths[vtx];
	const Scalar norm_f = lengths[vtx + 1];

	const Vec3x& te = tangents[vtx];
	const Vec3x& tf = tangents[vtx + 1];

	const Vec3x& m1e = materialFrames1[vtx];
	const Vec3x& m2e = materialFrames2[vtx];
	const Vec3x& m1f = materialFrames1[vtx + 1];
	const Vec3x& m2f = materialFrames2[vtx + 1];

	Scalar chi = MAX(1e-12, 1.0 + te.dot(tf));

	const Vec3x& tilde_t = (te + tf) / chi;
	const Vec3x& tilde_d1e = (2.0 * m1e) / chi;
	const Vec3x& tilde_d1f = (2.0 * m1f) / chi;
	const Vec3x& tilde_d2e = (2.0 * m2e) / chi;
	const Vec3x& tilde_d2f = (2.0 * m2f) / chi;

	const Vec4x& kappa = kappas[vtx];

	const Vec3x& Dkappa0eDe = 1.0 / norm_e * (-kappa(0) * tilde_t + tf.cross(tilde_d2e));
	const Vec3x& Dkappa0eDf = 1.0 / norm_f * (-kappa(0) * tilde_t - te.cross(tilde_d2e));
	const Vec3x& Dkappa1eDe = 1.0 / norm_e * (-kappa(1) * tilde_t - tf.cross(tilde_d1e));
	const Vec3x& Dkappa1eDf = 1.0 / norm_f * (-kappa(1) * tilde_t + te.cross(tilde_d1e));

	const Vec3x& Dkappa0fDe = 1.0 / norm_e * (-kappa(2) * tilde_t + tf.cross(tilde_d2f));
	const Vec3x& Dkappa0fDf = 1.0 / norm_f * (-kappa(2) * tilde_t - te.cross(tilde_d2f));
	const Vec3x& Dkappa1fDe = 1.0 / norm_e * (-kappa(3) * tilde_t - tf.cross(tilde_d1f));
	const Vec3x& Dkappa1fDf = 1.0 / norm_f * (-kappa(3) * tilde_t + te.cross(tilde_d1f));

	gradKappa.assignAt(0, 0, -Dkappa0eDe);
	gradKappa.assignAt(4, 0, Dkappa0eDe - Dkappa0eDf);
	gradKappa.assignAt(8, 0, Dkappa0eDf);
	gradKappa.assignAt(0, 1, -Dkappa1eDe);
	gradKappa.assignAt(4, 1, Dkappa1eDe - Dkappa1eDf);
	gradKappa.assignAt(8, 1, Dkappa1eDf);

	gradKappa.assignAt(0, 2, -Dkappa0fDe);
	gradKappa.assignAt(4, 2, Dkappa0fDe - Dkappa0fDf);
	gradKappa.assignAt(8, 2, Dkappa0fDf);
	gradKappa.assignAt(0, 3, -Dkappa1fDe);
	gradKappa.assignAt(4, 3, Dkappa1fDe - Dkappa1fDf);
	gradKappa.assignAt(8, 3, Dkappa1fDf);

	const Vec3x& kb = curvatureBinormals[vtx];

	gradKappa(3, 0) = -kb.dot(m1e);
	gradKappa(7, 0) = 0.0;
	gradKappa(3, 1) = -kb.dot(m2e);
	gradKappa(7, 1) = 0.0;

	gradKappa(3, 2) = 0.0;
	gradKappa(7, 2) = -kb.dot(m1f);
	gradKappa(3, 3) = 0.0;
	gradKappa(7, 3) = -kb.dot(m2f);
}

__global__ void computeHessKappa(Mat11x* hessKappas, const Scalar* lengths, const Vec3x* tangents, const Vec3x* curvatureBinormals,
	const Vec3x* materialFrames1, const Vec3x* materialFrames2, const Vec4x* kappas)
{
	int vtx = threadIdx.x;

	Mat11x& DDkappa0 = hessKappas[vtx * 4 + 0];
	Mat11x& DDkappa1 = hessKappas[vtx * 4 + 1];
	Mat11x& DDkappa2 = hessKappas[vtx * 4 + 2];
	Mat11x& DDkappa3 = hessKappas[vtx * 4 + 3];

	DDkappa0.setZero();
	DDkappa1.setZero();
	DDkappa2.setZero();
	DDkappa3.setZero();

	const Scalar norm_e = lengths[vtx];
	const Scalar norm_f = lengths[vtx + 1];
	const Scalar norm2_e = SQRT(norm_e); // That's bloody stupid, taking the square of a square root.
	const Scalar norm2_f = SQRT(norm_f);

	const Vec3x& te = tangents[vtx];
	const Vec3x& tf = tangents[vtx + 1];

	const Vec3x& m1e = materialFrames1[vtx];
	const Vec3x& m2e = materialFrames2[vtx];
	const Vec3x& m1f = materialFrames1[vtx + 1];
	const Vec3x& m2f = materialFrames2[vtx + 1];

	Scalar chi = MAX(1e-12, 1.0 + te.dot(tf));

	const Vec3x& tilde_t = (te + tf) / chi;
	const Vec3x& tilde_d1e = (2.0 * m1e) / chi;
	const Vec3x& tilde_d2e = (2.0 * m2e) / chi;
	const Vec3x& tilde_d1f = (2.0 * m1f) / chi;
	const Vec3x& tilde_d2f = (2.0 * m2f) / chi;

	const Vec4x& kappa = kappas[vtx];

	const Vec3x& Dkappa0eDe = 1.0 / norm_e * (-kappa(0) * tilde_t + tf.cross(tilde_d2e));
	const Vec3x& Dkappa0eDf = 1.0 / norm_f * (-kappa(0) * tilde_t - te.cross(tilde_d2e));
	const Vec3x& Dkappa1eDe = 1.0 / norm_e * (-kappa(1) * tilde_t - tf.cross(tilde_d1e));
	const Vec3x& Dkappa1eDf = 1.0 / norm_f * (-kappa(1) * tilde_t + te.cross(tilde_d1e));

	const Vec3x& Dkappa0fDe = 1.0 / norm_e * (-kappa(2) * tilde_t + tf.cross(tilde_d2f));
	const Vec3x& Dkappa0fDf = 1.0 / norm_f * (-kappa(2) * tilde_t - te.cross(tilde_d2f));
	const Vec3x& Dkappa1fDe = 1.0 / norm_e * (-kappa(3) * tilde_t - tf.cross(tilde_d1f));
	const Vec3x& Dkappa1fDf = 1.0 / norm_f * (-kappa(3) * tilde_t + te.cross(tilde_d1f));

	const Vec3x& kb = curvatureBinormals[vtx];

	const Mat3x& Id = Mat3x::Identity();

	const Vec3x& DchiDe = 1.0 / norm_e * (Id - outerProd(te, te)) * tf;
	const Vec3x& DchiDf = 1.0 / norm_f * (Id - outerProd(tf, tf)) * te;

	const Mat3x& DtfDf = 1.0 / norm_f * (Id - outerProd(tf, tf));

	const Mat3x& DttDe = 1.0 / (chi * norm_e) * ((Id - outerProd(te, te)) - outerProd(tilde_t, (Id - outerProd(te, te)) * tf));

	const Mat3x& DttDf = 1.0 / (chi * norm_f) * ((Id - outerProd(tf, tf)) - outerProd(tilde_t, (Id - outerProd(tf, tf)) * te));

	// 1st Hessian
	const Mat3x& D2kappa0De2 = -1.0 / norm_e * symPart(outerProd(tilde_t + te, Dkappa0eDe) + kappa(0) * DttDe + outerProd(1.0 / chi * tf.cross(tilde_d2e), DchiDe));

	const Mat3x& D2kappa0Df2 = -1.0 / norm_f * symPart(outerProd(tilde_t + tf, Dkappa0eDf) + kappa(0) * DttDf + outerProd(1.0 / chi * te.cross(tilde_d2e), DchiDf));

	const Mat3x& D2kappa0DeDf = -1.0 / norm_e * (outerProd(tilde_t, Dkappa0eDf) + kappa(0) * DttDf + outerProd(1.0 / chi * tf.cross(tilde_d2e), DchiDf) + crossMat(tilde_d2e) * DtfDf);

	const Mat3x& D2kappa0DfDe = D2kappa0DeDf.transpose();

	const Scalar D2kappa0Dthetae2 = -kb.dot(m2e);
	const Scalar D2kappa0Dthetaf2 = 0.0;
	const Vec3x& D2kappa0DeDthetae = 1.0 / norm_e * (kb.dot(m1e) * tilde_t - 2.0 / chi * tf.cross(m1e));
	const Vec3x& D2kappa0DeDthetaf = Vec3x::Zero();
	const Vec3x& D2kappa0DfDthetae = 1.0 / norm_f * (kb.dot(m1e) * tilde_t + 2.0 / chi * te.cross(m1e));
	const Vec3x& D2kappa0DfDthetaf = Vec3x::Zero();

	// 2nd Hessian
	const Mat3x& D2kappa1De2 = -1.0 / norm_e * symPart(outerProd(tilde_t + te, Dkappa1eDe) + kappa(1) * DttDe + outerProd(1.0 / chi * tf.cross(tilde_d1e), DchiDe));

	const Mat3x& D2kappa1Df2 = -1.0 / norm_f * symPart(outerProd(tilde_t + tf, Dkappa1eDf) + kappa(1) * DttDf + outerProd(1.0 / chi * te.cross(tilde_d1e), DchiDf));

	const Mat3x& D2kappa1DeDf = -1.0 / norm_e * (outerProd(tilde_t, Dkappa1eDf) + kappa(1) * DttDf - outerProd(1.0 / chi * tf.cross(tilde_d1e), DchiDf) - crossMat(tilde_d1e) * DtfDf);

	const Mat3x& D2kappa1DfDe = D2kappa1DeDf.transpose();

	const Scalar D2kappa1Dthetae2 = kb.dot(m1e);
	const Scalar D2kappa1Dthetaf2 = 0.0;
	const Vec3x& D2kappa1DeDthetae = 1.0 / norm_e * (kb.dot(m2e) * tilde_t - 2.0 / chi * tf.cross(m2e));
	const Vec3x& D2kappa1DeDthetaf = Vec3x::Zero();
	const Vec3x& D2kappa1DfDthetae = 1.0 / norm_f * (kb.dot(m2e) * tilde_t + 2.0 / chi * te.cross(m2e));
	const Vec3x& D2kappa1DfDthetaf = Vec3x::Zero();

	// 3rd Hessian
	const Mat3x& D2kappa2De2 = -1.0 / norm_e * symPart(outerProd(tilde_t + te, Dkappa0fDe) + kappa(2) * DttDe + outerProd(1.0 / chi * tf.cross(tilde_d2f), DchiDe));

	const Mat3x& D2kappa2Df2 = -1.0 / norm_f * symPart(outerProd(tilde_t + tf, Dkappa0fDf) + kappa(2) * DttDf + outerProd(1.0 / chi * te.cross(tilde_d2f), DchiDf));

	const Mat3x& D2kappa2DeDf = -1.0 / norm_e * (outerProd(tilde_t, Dkappa0fDf) + kappa(2) * DttDf + outerProd(1.0 / chi * tf.cross(tilde_d2f), DchiDf) + crossMat(tilde_d2f) * DtfDf);

	const Mat3x& D2kappa2DfDe = D2kappa2DeDf.transpose();

	const Scalar D2kappa2Dthetae2 = 0.0;
	const Scalar D2kappa2Dthetaf2 = -kb.dot(m2f);
	const Vec3x& D2kappa2DeDthetae = Vec3x::Zero();
	const Vec3x& D2kappa2DeDthetaf = 1.0 / norm_e * (kb.dot(m1f) * tilde_t - 2.0 / chi * tf.cross(m1f));
	const Vec3x& D2kappa2DfDthetae = Vec3x::Zero();
	const Vec3x& D2kappa2DfDthetaf = 1.0 / norm_f * (kb.dot(m1f) * tilde_t + 2.0 / chi * te.cross(m1f));

	// 4th Hessian
	const Mat3x& D2kappa3De2 = -1.0 / norm_e * symPart(outerProd(tilde_t + te, Dkappa1fDe) + kappa(3) * DttDe + outerProd(1.0 / chi * tf.cross(tilde_d1f), DchiDe));

	const Mat3x& D2kappa3Df2 = -1.0 / norm_f * symPart(outerProd(tilde_t + tf, Dkappa1fDf) + kappa(3) * DttDf + outerProd(1.0 / chi * te.cross(tilde_d1f), DchiDf));

	const Mat3x& D2kappa3DeDf = -1.0 / norm_e * (outerProd(tilde_t, Dkappa1fDf) + kappa(3) * DttDf - outerProd(1.0 / chi * tf.cross(tilde_d1f), DchiDf) - crossMat(tilde_d1f) * DtfDf);

	const Mat3x& D2kappa3DfDe = D2kappa3DeDf.transpose();

	const Scalar D2kappa3Dthetae2 = 0.0;
	const Scalar D2kappa3Dthetaf2 = kb.dot(m1f);
	const Vec3x& D2kappa3DeDthetae = Vec3x::Zero();
	const Vec3x& D2kappa3DeDthetaf = 1.0 / norm_e * (kb.dot(m2f) * tilde_t - 2.0 / chi * tf.cross(m2f));
	const Vec3x& D2kappa3DfDthetae = Vec3x::Zero();
	const Vec3x& D2kappa3DfDthetaf = 1.0 / norm_f * (kb.dot(m2f) * tilde_t + 2.0 / chi * te.cross(m2f));

	DDkappa0.assignAt(0, 0, D2kappa0De2);
	DDkappa0.assignAt(0, 4, -D2kappa0De2 + D2kappa0DeDf);
	DDkappa0.assignAt(4, 0, -D2kappa0De2 + D2kappa0DfDe);
	DDkappa0.assignAt(4, 4, D2kappa0De2 - (D2kappa0DeDf + D2kappa0DfDe) + D2kappa0Df2);
	DDkappa0.assignAt(0, 8, -D2kappa0DeDf);
	DDkappa0.assignAt(8, 0, -D2kappa0DfDe);
	DDkappa0.assignAt(4, 8, D2kappa0DeDf - D2kappa0Df2);
	DDkappa0.assignAt(8, 4, D2kappa0DfDe - D2kappa0Df2);
	DDkappa0.assignAt(8, 8, D2kappa0Df2);
	DDkappa0(3, 3) = D2kappa0Dthetae2;
	DDkappa0(7, 7) = D2kappa0Dthetaf2;
	DDkappa0(3, 7) = DDkappa0(7, 3) = 0.;
	DDkappa0.assignAt<true>(0, 3, -D2kappa0DeDthetae);
	DDkappa0.assignAt<false>(3, 0, -D2kappa0DeDthetae);
	DDkappa0.assignAt<true>(4, 3, D2kappa0DeDthetae - D2kappa0DfDthetae);
	DDkappa0.assignAt<false>(3, 4, D2kappa0DeDthetae - D2kappa0DfDthetae);
	DDkappa0.assignAt<true>(8, 3, D2kappa0DfDthetae);
	DDkappa0.assignAt<false>(3, 8, D2kappa0DfDthetae);
	DDkappa0.assignAt<true>(0, 7, -D2kappa0DeDthetaf);
	DDkappa0.assignAt<false>(7, 0, -D2kappa0DeDthetaf);
	DDkappa0.assignAt<true>(4, 7, D2kappa0DeDthetaf - D2kappa0DfDthetaf);
	DDkappa0.assignAt<false>(7, 4, D2kappa0DeDthetaf - D2kappa0DfDthetaf);
	DDkappa0.assignAt<true>(8, 7, D2kappa0DfDthetaf);
	DDkappa0.assignAt<false>(7, 8, D2kappa0DfDthetaf);

	assert(isSymmetric(DDkappa0));

	DDkappa1.assignAt(0, 0, D2kappa1De2);
	DDkappa1.assignAt(0, 4, -D2kappa1De2 + D2kappa1DeDf);
	DDkappa1.assignAt(4, 0, -D2kappa1De2 + D2kappa1DfDe);
	DDkappa1.assignAt(4, 4, D2kappa1De2 - (D2kappa1DeDf + D2kappa1DfDe) + D2kappa1Df2);
	DDkappa1.assignAt(0, 8, -D2kappa1DeDf);
	DDkappa1.assignAt(8, 0, -D2kappa1DfDe);
	DDkappa1.assignAt(4, 8, D2kappa1DeDf - D2kappa1Df2);
	DDkappa1.assignAt(8, 4, D2kappa1DfDe - D2kappa1Df2);
	DDkappa1.assignAt(8, 8, D2kappa1Df2);
	DDkappa1(3, 3) = D2kappa1Dthetae2;
	DDkappa1(7, 7) = D2kappa1Dthetaf2;
	DDkappa1(3, 7) = DDkappa1(7, 3) = 0.;
	DDkappa1.assignAt<true>(0, 3, -D2kappa1DeDthetae);
	DDkappa1.assignAt<false>(3, 0, -D2kappa1DeDthetae);
	DDkappa1.assignAt<true>(4, 3, D2kappa1DeDthetae - D2kappa1DfDthetae);
	DDkappa1.assignAt<false>(3, 4, D2kappa1DeDthetae - D2kappa1DfDthetae);
	DDkappa1.assignAt<true>(8, 3, D2kappa1DfDthetae);
	DDkappa1.assignAt<false>(3, 8, D2kappa1DfDthetae);
	DDkappa1.assignAt<true>(0, 7, -D2kappa1DeDthetaf);
	DDkappa1.assignAt<false>(7, 0, -D2kappa1DeDthetaf);
	DDkappa1.assignAt<true>(4, 7, D2kappa1DeDthetaf - D2kappa1DfDthetaf);
	DDkappa1.assignAt<false>(7, 4, D2kappa1DeDthetaf - D2kappa1DfDthetaf);
	DDkappa1.assignAt<true>(8, 7, D2kappa1DfDthetaf);
	DDkappa1.assignAt<false>(7, 8, D2kappa1DfDthetaf);

	assert(isSymmetric(DDkappa1));

	DDkappa2.assignAt(0, 0, D2kappa2De2);
	DDkappa2.assignAt(0, 4, -D2kappa2De2 + D2kappa2DeDf);
	DDkappa2.assignAt(4, 0, -D2kappa2De2 + D2kappa2DfDe);
	DDkappa2.assignAt(4, 4, D2kappa2De2 - (D2kappa2DeDf + D2kappa2DfDe) + D2kappa2Df2);
	DDkappa2.assignAt(0, 8, -D2kappa2DeDf);
	DDkappa2.assignAt(8, 0, -D2kappa2DfDe);
	DDkappa2.assignAt(4, 8, D2kappa2DeDf - D2kappa2Df2);
	DDkappa2.assignAt(8, 4, D2kappa2DfDe - D2kappa2Df2);
	DDkappa2.assignAt(8, 8, D2kappa2Df2);
	DDkappa2(3, 3) = D2kappa2Dthetae2;
	DDkappa2(7, 7) = D2kappa2Dthetaf2;
	DDkappa2(3, 7) = DDkappa2(7, 3) = 0.;
	DDkappa2.assignAt<true>(0, 3, -D2kappa2DeDthetae);
	DDkappa2.assignAt<false>(3, 0, -D2kappa2DeDthetae);
	DDkappa2.assignAt<true>(4, 3, D2kappa2DeDthetae - D2kappa2DfDthetae);
	DDkappa2.assignAt<false>(3, 4, D2kappa2DeDthetae - D2kappa2DfDthetae);
	DDkappa2.assignAt<true>(8, 3, D2kappa2DfDthetae);
	DDkappa2.assignAt<false>(3, 8, D2kappa2DfDthetae);
	DDkappa2.assignAt<true>(0, 7, -D2kappa2DeDthetaf);
	DDkappa2.assignAt<false>(7, 0, -D2kappa2DeDthetaf);
	DDkappa2.assignAt<true>(4, 7, D2kappa2DeDthetaf - D2kappa2DfDthetaf);
	DDkappa2.assignAt<false>(7, 4, D2kappa2DeDthetaf - D2kappa2DfDthetaf);
	DDkappa2.assignAt<true>(8, 7, D2kappa2DfDthetaf);
	DDkappa2.assignAt<false>(7, 8, D2kappa2DfDthetaf);

	assert(isSymmetric(DDkappa2));

	DDkappa3.assignAt(0, 0, D2kappa3De2);
	DDkappa3.assignAt(0, 4, -D2kappa3De2 + D2kappa3DeDf);
	DDkappa3.assignAt(4, 0, -D2kappa3De2 + D2kappa3DfDe);
	DDkappa3.assignAt(4, 4, D2kappa3De2 - (D2kappa3DeDf + D2kappa3DfDe) + D2kappa3Df2);
	DDkappa3.assignAt(0, 8, -D2kappa3DeDf);
	DDkappa3.assignAt(8, 0, -D2kappa3DfDe);
	DDkappa3.assignAt(4, 8, D2kappa3DeDf - D2kappa3Df2);
	DDkappa3.assignAt(8, 4, D2kappa3DfDe - D2kappa3Df2);
	DDkappa3.assignAt(8, 8, D2kappa3Df2);
	DDkappa3(3, 3) = D2kappa3Dthetae2;
	DDkappa3(7, 7) = D2kappa3Dthetaf2;
	DDkappa3(3, 7) = DDkappa3(7, 3) = 0.;
	DDkappa3.assignAt<true>(0, 3, -D2kappa3DeDthetae);
	DDkappa3.assignAt<false>(3, 0, -D2kappa3DeDthetae);
	DDkappa3.assignAt<true>(4, 3, D2kappa3DeDthetae - D2kappa3DfDthetae);
	DDkappa3.assignAt<false>(3, 4, D2kappa3DeDthetae - D2kappa3DfDthetae);
	DDkappa3.assignAt<true>(8, 3, D2kappa3DfDthetae);
	DDkappa3.assignAt<false>(3, 8, D2kappa3DfDthetae);
	DDkappa3.assignAt<true>(0, 7, -D2kappa3DeDthetaf);
	DDkappa3.assignAt<false>(7, 0, -D2kappa3DeDthetaf);
	DDkappa3.assignAt<true>(4, 7, D2kappa3DeDthetaf - D2kappa3DfDthetaf);
	DDkappa3.assignAt<false>(7, 4, D2kappa3DeDthetaf - D2kappa3DfDthetaf);
	DDkappa3.assignAt<true>(8, 7, D2kappa3DfDthetaf);
	DDkappa3.assignAt<false>(7, 8, D2kappa3DfDthetaf);

	assert(isSymmetric(DDkappa3));
}

__global__ void computeBendingEnergy(Scalar kb, Scalar* bendingEnergy, const Vec4x* kappas, const Vec4x* restKappas, const Scalar* ilens)
{
	int i = threadIdx.x;

	extern __shared__ Scalar energy[];
	energy[i] = 0.25 * kb * ilens[i + 1] * (kappas[i] - restKappas[i]).norm2();
	__syncthreads();

	// Reduction
	int length = blockDim.x;
	int strip = (length + 1) / 2;
	while (length > 1)
	{
		if (i < strip && i + strip < length)
		{
			energy[i] += energy[i + strip];
		}
		__syncthreads();
		length = strip;
		strip = (length + 1) / 2;
	}

	bendingEnergy[i] = energy[i];
}

__global__ void computeBendingForce(Scalar kb, Scalar* totalForces, const Vec4x* kappas, const Vec4x* restKappas, 
	const Scalar* ilens, const Mat11x4x* gradKappas)
{
	int i = threadIdx.x;

	Vec11x force = -0.5 * kb * ilens[i + 1] * gradKappas[i] * (kappas[i] - restKappas[i]);

	int j;
	for (j = 0; j < 4; ++j)
		totalForces[4 * i + j] += force(j);
	__syncthreads();
	for (j = 4; j < 8; ++j)
		totalForces[4 * i + j] += force(j);
	__syncthreads();
	for (j = 8; j < 11; ++j)
		totalForces[4 * i + j] += force(j);
}

__global__ void computeBendingGradient(Scalar kb, MatrixWrapper<Scalar>* gradient, const Vec4x* kappas, const Vec4x* restKappas,
	const Scalar* ilens, const Mat11x4x* gradKappas, const Mat11x* hessKappas)
{
	int i = threadIdx.x;

	Mat11x localJ = symProd(gradKappas[i]);
	Vec4x deltaKappa = kappas[i] - restKappas[i];
	for (int idx = 0; idx < 4; ++idx)
		localJ += hessKappas[4 * i + idx] * deltaKappa(idx);
	localJ *= 0.5 * kb * ilens[i + 1];

	int j, k;
	for (j = 0; j < 4; ++j) // J11, J12, J13
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
	__syncthreads();
	for (j = 4; j < 8; ++j) // J21, J22, J23
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
	__syncthreads();
	for (j = 8; j < 11; ++j) // J31, J32, J33
	{
		for (k = 0; k < 11; ++k) {
			gradient->addInPlace(4 * i + j, 4 * i + k, localJ(j, k));
		}
	}
}

StrandForces::StrandForces(const ElasticStrand* strand, cudaStream_t stream) :
	m_strand(strand),
	m_params(strand->getParameters()),
	m_numVertices(strand->getNumVertices()),
	m_numEdges(strand->getNumEdges()),
	m_stream(stream),
	m_totalEnergy(0.),
	m_totalGradient(strand->getNumDof(), strand->getStream()),
	m_state(strand->getNumVertices(), stream)
{
	// Alocate device memory
	cudaMalloc((void**)& m_fixingEnergy, m_numVertices * sizeof(Scalar));
	cudaMalloc((void**)& m_stretchingEnergy, m_numEdges * sizeof(Scalar));
	cudaMalloc((void**)& m_twistingEnergy, (m_numVertices - 2) * sizeof(Scalar));
	cudaMalloc((void**)& m_bendingEnergy, (m_numVertices - 2) * sizeof(Scalar));

	cudaMalloc((void**)& m_gradTwists, (m_numVertices - 2) * sizeof(Vec11x));
	cudaMalloc((void**)& m_hessTwists, (m_numVertices - 2) * sizeof(Mat11x));
	cudaMalloc((void**)& m_gradKappas, (m_numVertices - 2) * sizeof(Mat11x4x));
	cudaMalloc((void**)& m_hessKappas, (m_numVertices - 2) * 4 * sizeof(Mat11x));

	cudaMalloc((void**)& m_totalForces, (4 * m_numVertices - 1) * sizeof(Scalar));
}

StrandForces::~StrandForces()
{
	cudaFree(m_fixingEnergy);
	cudaFree(m_stretchingEnergy);
	cudaFree(m_twistingEnergy);
	cudaFree(m_bendingEnergy);

	cudaFree(m_gradTwists);
	cudaFree(m_hessTwists);
	cudaFree(m_gradKappas);
	cudaFree(m_hessKappas);

	cudaFree(m_totalForces);
}

void StrandForces::computeForceAndGradient()
{
	// Assign 0
	set<Scalar> <<< 1, m_strand->getNumDof(), 0, m_stream >>> (m_strand->getNumDof(), m_totalForces, 0.0);
	m_totalGradient.setZero();

	m_state.update(m_strand->getCurrentState());

	computeFixing();
	computeStretching();
	computeTwisting();
	computeBending();
}

void StrandForces::computeFixing()
{
	int n_fixed = m_strand->getNumFixed();
	const int* indices = m_strand->getFixedIndices();
	const Vec3x* positions = m_strand->getFixedPositions();
	Scalar ks = m_params->m_ks;

	computeFixingForce <<< 1, n_fixed, 0, m_stream >>> (ks, m_totalForces, indices, m_state.m_x, positions);
	computeFixingGradient <<< 1, n_fixed, 0, m_stream >>> (ks, m_totalGradient.getMatrix(), indices);
}

void StrandForces::computeStretching()
{
	Scalar ks = m_params->m_ks * m_params->m_stretchMultiplier;
	const Scalar* lengths = m_state.m_lengths;
	const Scalar* restLengths = m_strand->getRestLengths();
	const Vec3x* tangents = m_state.m_tangents;

	computeStretchingForces <<< 1, m_numEdges, 0, m_stream >>> (ks, m_totalForces, lengths, restLengths, tangents);
	computeStretchingGradient <<< 1, m_numEdges, 0, m_stream >>> (ks, m_totalGradient.getMatrix(), lengths, restLengths, tangents);
}

void StrandForces::computeTwisting()
{
	Scalar kt = m_params->m_kt;
	const Scalar* restTwists = m_strand->getRestTwists();
	const Scalar* ilens = m_strand->getInvVtxLengths();
	const Scalar* lengths = m_state.m_lengths;
	const Vec3x* tangents = m_state.m_tangents;
	const Vec3x* curvatureBinormals = m_state.m_curvatureBinormals;
	const Scalar* twists = m_state.m_twists;

	int nt = m_numVertices - 2;

	computeGradTwist <<< 1, nt, 0, m_stream >>> (m_gradTwists, curvatureBinormals, lengths);
	computeHessTwist <<< 1, nt, 0, m_stream >>> (m_hessTwists, tangents, lengths, curvatureBinormals);
	computeTwistingForce <<< 1, nt, 0, m_stream >>> (kt, m_totalForces, twists, restTwists, ilens, m_gradTwists);
	computeTwistingGradient <<< 1, nt, 0, m_stream >>> (kt, m_totalGradient.getMatrix(), twists, restTwists, ilens, m_gradTwists, m_hessTwists);
}

void StrandForces::computeBending()
{
	Scalar kb = m_params->m_kb;
	const Scalar* lengths = m_state.m_lengths;
	const Scalar* ilens = m_strand->getInvVtxLengths();
	const Vec3x* tangents = m_state.m_tangents;
	const Vec3x* mat1 = m_state.m_materialFrames1;
	const Vec3x* mat2 = m_state.m_materialFrames2;
	const Vec3x* curvatureBinormals = m_state.m_curvatureBinormals;
	const Vec4x* kappas = m_state.m_kappas;
	const Vec4x* restKappas = m_strand->getRestKappas();

	int nb = m_numVertices - 2;

	computeGradKappa << < 1, nb, 0, m_stream >> > (m_gradKappas, lengths, tangents, curvatureBinormals, mat1, mat2, kappas);
	computeHessKappa << < 1, nb, 0, m_stream >> > (m_hessKappas, lengths, tangents, curvatureBinormals, mat1, mat2, kappas);

	computeBendingForce << < 1, nb, 0, m_stream >> > (kb, m_totalForces, kappas, restKappas, ilens, m_gradKappas);
	computeBendingGradient << < 1, nb, 0, m_stream >> > (kb, m_totalGradient.getMatrix(), kappas, restKappas, ilens, m_gradKappas, m_hessKappas);
}

void StrandForces::computeEnergy()
{
	m_totalEnergy = 0.0;
	m_state.update(m_strand->getCurrentState());

	Scalar energy;

	// Stretching
	computeStretchingEnergy <<< 1, m_numEdges, m_numEdges * sizeof(Scalar), m_stream >>> (
		m_params->m_ks * m_params->m_stretchMultiplier, m_stretchingEnergy, m_state.m_lengths, m_strand->getRestLengths());
	cudaMemcpyAsync(&energy, m_stretchingEnergy, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	m_totalEnergy += energy;

	// Fixing
	int n_fixed = m_strand->getNumFixed();
	computeFixingEnergy <<< 1,n_fixed, n_fixed * sizeof(Scalar), m_stream >>> (m_params->m_ks, m_fixingEnergy, 
		m_strand->getFixedIndices(), m_state.m_x, m_strand->getFixedPositions());
	cudaMemcpyAsync(&energy, m_fixingEnergy, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	m_totalEnergy += energy;

	computeTwistingEnergy <<< 1, m_numVertices - 2, (m_numVertices - 2) * sizeof(Scalar), m_stream >>> 
		(m_params->m_kt, m_twistingEnergy, m_state.m_twists, m_strand->getRestTwists(), m_strand->getInvVtxLengths());
	cudaMemcpyAsync(&energy, m_twistingEnergy, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	m_totalEnergy += energy;

	computeBendingEnergy <<< 1, m_numVertices - 2, (m_numVertices - 2) * sizeof(Scalar), m_stream >> >
		(m_params->m_kb, m_bendingEnergy, m_state.m_kappas, m_strand->getRestKappas(), m_strand->getInvVtxLengths());
	cudaMemcpyAsync(&energy, m_bendingEnergy, sizeof(Scalar), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	m_totalEnergy += energy;
}
