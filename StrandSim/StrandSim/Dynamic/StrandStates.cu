#include "StrandStates.h"
#include "../Utils/CUDAMathDef.h"
#include "BandMatrix.h"

#include <iostream>
#include "../Utils/EigenDef.h"

__global__ void updateEdges(int ne, const int* edge_to_strand, const Scalar* x, Scalar* lengths, Vec3x* tangents)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ne) return;

	int idx = 4 * i + 3 * edge_to_strand[i];
	Vec3x edge(x[idx + 4] - x[idx], x[idx + 5] - x[idx + 1], x[idx + 6] - x[idx + 2]);

	lengths[i] = edge.norm();
	tangents[i] = edge.normalized();
}

__global__ void updateFrames(int ne, const int* edge_to_strand, const Scalar* x, const Vec3x* last_ref1, const Vec3x* last_tangents, 
	const Vec3x* tangents, Vec3x* ref1, Vec3x* mat1, Vec3x* mat2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ne) return;

	Vec3x u = orthonormalParallelTransport(last_ref1[i], last_tangents[i], tangents[i]);
	Vec3x v = tangents[i].cross(ref1[i]);

	int j = 4 * i + 3 * edge_to_strand[i] + 3;
	Scalar s = sin(x[j]);
	Scalar c = cos(x[j]);

	ref1[i] = u;
	mat1[i] = c * u + s * v;
	mat2[i] = -s * u + c * v;
}

__global__ void updateTwists(int nt, const int* ivtx_to_strand, const Scalar* x, const Vec3x* tangents, const Vec3x* ref1, Scalar* twists)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	int j = i + ivtx_to_strand[i];
	int k = 4 * i + 7 * ivtx_to_strand[i];

	Vec3x u = orthonormalParallelTransport(ref1[j], tangents[j], tangents[j + 1]);
	Scalar refTwist = signedAngle(u, ref1[j + 1], tangents[j + 1]);

	twists[i] = refTwist + (x[k + 7] - x[k + 3]);
}

__global__ void updateKappas(int nb, const int* ivtx_to_strand, const Scalar* x, const Vec3x* tangents, const Vec3x* mat1, 
	const Vec3x* mat2, Vec3x* curvatureBinormals, Vec4x* kappas)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	int j = i + ivtx_to_strand[i];

	const Vec3x& t0 = tangents[j];
	const Vec3x& t1 = tangents[j + 1];
	const Vec3x& m11 = mat1[j];
	const Vec3x& m12 = mat2[j];
	const Vec3x& m21 = mat1[j + 1];
	const Vec3x& m22 = mat2[j + 1];

	Scalar denominator = MAX(1e-12, 1.0 + t0.dot(t1));
	Vec3x kb = 2.0 * t0.cross(t1) / denominator;
	curvatureBinormals[i] = kb;
	kappas[i] = Vec4x(kb.dot(m12), -kb.dot(m11), kb.dot(m22), -kb.dot(m21));
}

__global__ void initMass(int nv, const int* strand_ptr, const int* vtx_to_strand, const Scalar* lengths, Scalar m_per_len, Scalar r, Scalar* mass)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nv) return;

	int sid = vtx_to_strand[i];
	int eid = i - sid;
	int vtx_idx = 4 * i - sid;

	Scalar mv, me;

	if (i == strand_ptr[sid])	// first vertex
	{
		mv = 0.5 * m_per_len * lengths[eid];
		mass[vtx_idx] = mv;
		mass[vtx_idx + 1] = mv;
		mass[vtx_idx + 2] = mv;
		mass[vtx_idx + 3] = mv * r * r;
		return;
	}
	else if (i == strand_ptr[sid + 1] - 1)	// last vertex
	{
		mv = 0.5 * m_per_len * lengths[eid - 1];
		mass[vtx_idx] = mv;
		mass[vtx_idx + 1] = mv;
		mass[vtx_idx + 2] = mv;
		return;
	}
	else
	{
		mv = 0.5 * m_per_len * (lengths[eid - 1] + lengths[eid]);
		me = 0.5 * m_per_len * lengths[eid] * r * r;
		mass[vtx_idx] = mv;
		mass[vtx_idx + 1] = mv;
		mass[vtx_idx + 2] = mv;
		mass[vtx_idx + 3] = me;
	}
}

__global__ void initGravity(int nv, const int* vtx_to_strand, const Scalar* mass, Scalar gy, Scalar* gravity)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nv) return;

	int j = 4 * i - vtx_to_strand[i] + 1;
	gravity[j] = gy * mass[j];
}

__global__ void initInvVtxLengths(int nv, const int* ivtx_to_strand, const Scalar* lengths, Scalar* invLen)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nv) return;

	int j = i + ivtx_to_strand[i];
	invLen[i] = 2 /  (lengths[j] + lengths[j + 1]);;
}

__global__ void initRefFrames(int ns, const int* strand_ptr, const Vec3x* tangents, Vec3x* ref1)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ns) return;

	int start_e = strand_ptr[i] - i;
	int end_e = strand_ptr[i + 1] - i - 1;

	// Computes reference frame of the first edge
	Vec3x t0 = tangents[start_e], t1;
	ref1[start_e] = findNormal(t0);

	for (int e = start_e + 1; e < end_e; ++e)
	{
		ref1[e] = orthonormalParallelTransport(ref1[e - 1], tangents[e - 1], tangents[e]);
	}
}

__global__ void initMatFrames(int ne, const int* edge_to_strand, const Scalar* x, const Vec3x* tangents, const Vec3x* ref1, 
	Vec3x* mat1, Vec3x* mat2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ne) return;

	int j = 4 * i + 3 * edge_to_strand[i] + 3;

	Vec3x u = ref1[i];
	Vec3x v = tangents[i].cross(u);
	Scalar s = sin(x[j]);
	Scalar c = cos(x[j]);

	mat1[i] = c * u + s * v;
	mat2[i] = -s * u + c * v;
}

__global__ void computeFixingEnergy(int nf, Scalar ks, Scalar* fixingEnergy, const int* fixed_idx, const int* vtx_to_strand, const Vec3x* target, const Scalar* x)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nf) return;

	int j = 4 * fixed_idx[i] - vtx_to_strand[fixed_idx[i]];
	Vec3x cur_vtx(x[j], x[j + 1], x[j + 2]);
	fixingEnergy[i] = 0.5 * ks * (cur_vtx - target[i]).norm2();
}

__global__ void computeFixingForces(int nf, Scalar ks, const int* fixed_idx, const int* vtx_to_strand, const Vec3x* target, const Scalar* x, Scalar* forces)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nf) return;

	int j = 4 * fixed_idx[i] - vtx_to_strand[fixed_idx[i]];

	forces[j] += ks * (target[i](0) - x[j]);
	forces[j + 1] += ks * (target[i](1) - x[j + 1]);
	forces[j + 2] += ks * (target[i](2) - x[j + 2]);
}

__global__ void computeFixingGradient(int nf, Scalar ks, const int* fixed_idx, const int* vtx_to_strand, MatrixWrapper<Scalar>* gradient)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nf) return;

	int j = 4 * fixed_idx[i] - vtx_to_strand[fixed_idx[i]];
	gradient->add(j, j, ks);
	gradient->add(j + 1, j + 1, ks);
	gradient->add(j + 2, j + 2, ks);
}

__global__ void computeFixingGradient(int w, int nf, Scalar ks, const int* strand_ptr, const int* fixed_idx, const int* vtx_to_strand, Scalar* gradient)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nf) return;

	int sid = vtx_to_strand[fixed_idx[i]];
	int n_dof = 4 * (strand_ptr[sid + 1] - strand_ptr[sid]) - 1;
	int start_dof = 4 * strand_ptr[sid] - sid;
	int base = start_dof * (2 * w + 1) + w * n_dof;

	int j = 4 * fixed_idx[i] - sid - start_dof;

	gradient[base + j] += ks;
	gradient[base + j + 1] += ks;
	gradient[base + j + 2] += ks;
}

__global__ void computeStretchingEnergy(int ns, Scalar ks, Scalar* stretchingEnergy, const Scalar* lengths, const Scalar* restLengths)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ns) return;

	stretchingEnergy[i] = 0.5 * ks * (lengths[i] - restLengths[i]) * (lengths[i] - restLengths[i]) / restLengths[i];
}

__global__ void computeStretchingForces(int ns, Scalar ks, const int* edge_to_strand, Scalar* forces, const Scalar* lengths, 
	const Scalar* restLengths, const Vec3x* tangents)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ns) return;

	Vec3x f = ks * (lengths[i] / restLengths[i] - 1) * tangents[i];

	Scalar* start_ptr = forces + 4 * i + 3 * edge_to_strand[i];

	atomicAdd(start_ptr, f(0));
	atomicAdd(start_ptr + 1, f(1));
	atomicAdd(start_ptr + 2, f(2));

	atomicAdd(start_ptr + 4, -f(0));
	atomicAdd(start_ptr + 5, -f(1));
	atomicAdd(start_ptr + 6, -f(2));
}

__global__ void computeStretchingForces(const int* strand_ptr, Scalar ks, const Scalar* lengths, const Scalar* restLengths, const Vec3x* tangents, Scalar* forces)
{
	int sid = blockIdx.x;
	int i = threadIdx.x + strand_ptr[sid] - sid;
	if (i >= strand_ptr[sid + 1] - sid - 1) return;

	Vec3x f = ks * (lengths[i] / restLengths[i] - 1) * tangents[i];

	int base = 4 * i + 3 * sid;

	forces[base] += f(0);
	forces[base + 1] += f(1);
	forces[base + 2] += f(2);
	__syncthreads();
	forces[base + 4] -= f(0);
	forces[base + 5] -= f(1);
	forces[base + 6] -= f(2);
}

__global__ void computeStretchingGradient(int ns, Scalar ks, const int* edge_to_strand, MatrixWrapper<Scalar>* gradient, const Scalar* lengths,
	const Scalar* restLengths, const Vec3x* tangents)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= ns) return;

	Mat3x M = ks * (1.0 / lengths[i] * outerProd(tangents[i]));
	if (lengths[i] > restLengths[i])
		M += ks * (1.0 / restLengths[i] - 1.0 / lengths[i]) * Mat3x::Identity();

	int idx = 4 * i + 3 * edge_to_strand[i];

	// Accumulte to hessian matrix
#pragma unroll
	for (int j = 0; j < 3; ++j)
	{
#pragma unroll
		for (int k = 0; k < 3; ++k)
		{
			gradient->add(idx + j, idx + k, M(j, k));
			gradient->add(idx + j, idx + 4 + k, -M(j, k));
			gradient->add(idx + 4 + j, idx + k, -M(j, k));
			gradient->add(idx + 4 + j, idx + 4 + k, M(j, k));
		}
	}
}

template <int w = BAND_WIDTH>
__global__ void computeStretchingGradient(const int* strand_ptr, Scalar ks, Scalar* band_matrix, const Scalar* lengths,
	const Scalar* restLengths, const Vec3x* tangents)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;
	int vid0 = strand_ptr[sid], vid1 = strand_ptr[sid + 1];
	int n_edge = vid1 - vid0 - 1;
	if (tid >= n_edge) return;

	extern __shared__ Scalar values[];
	// global -> shared
	int A_base = (4 * vid0 - sid) * (2 * w + 1);
	int n_dof = 4 * (vid1 - vid0) - 1;
	BandMatrixWrapper<> A(values, n_dof);
	int i, j, idx;
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			A[idx + j] = band_matrix[A_base + idx + j];
		}
		if (tid == n_edge - 1) {
			A[idx + 4] = band_matrix[A_base + idx + 4];
			A[idx + 5] = band_matrix[A_base + idx + 5];
			A[idx + 6] = band_matrix[A_base + idx + 6];
		}
	}

	idx = tid + vid0 - sid;  // global edge idx
	Mat3x M = ks * (1.0 / lengths[idx] * outerProd(tangents[idx]));
	if (lengths[idx] > restLengths[idx])
		M += ks * (1.0 / restLengths[idx] - 1.0 / lengths[idx]) * Mat3x::Identity();

	int base = 4 * tid;  // local dof idx
#pragma unroll
	for (i = 0; i < 3; ++i) {
#pragma unroll
		for (j = 0; j < 3; ++j) {
			A(base + i, base + j) += M(i, j);
			A(base + i, base + j + 4) -= M(i, j);
			A(base + i + 4, base + j) -= M(i, j);
		}
	}
	__syncthreads();
#pragma unroll
	for (i = 0; i < 3; ++i) {
#pragma unroll
		for (j = 0; j < 3; ++j) {
			A(base + i + 4, base + j + 4) += M(i, j);
		}
	}

	// shared -> global
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			band_matrix[A_base + idx + j] = A[idx + j];
		}
		if (tid == n_edge - 1) {
			band_matrix[A_base + idx + 4] = A[idx + 4];
			band_matrix[A_base + idx + 5] = A[idx + 5];
			band_matrix[A_base + idx + 6] = A[idx + 6];
		}
	}
}

__global__ void computeGradTwist(int nt, const int* ivtx_to_strand, Vec11x* gradTwists, const Vec3x* curvatureBinormals, const Scalar* lengths)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	int j = i + ivtx_to_strand[i];

	Vec11x& Dtwist = gradTwists[i];
	const Vec3x& kb = curvatureBinormals[i];

	Dtwist.assignAt(0, -0.5 / lengths[j] * kb);
	Dtwist.assignAt(8, 0.5 / lengths[j + 1] * kb);
	Dtwist.assignAt(4, -(Dtwist.at<3>(0) + Dtwist.at<3>(8)));
	Dtwist(3) = -1;
	Dtwist(7) = 1;
}

__global__ void computeHessTwist(int nt, const int* ivtx_to_strand, Mat11x* hessTwists, const Vec3x* tangents, 
	const Scalar* lengths, const Vec3x* curvatureBinormals)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	Mat11x DDtwist;

	int j = i + ivtx_to_strand[i];

	const Vec3x& te = tangents[j];
	const Vec3x& tf = tangents[j + 1];
	const Scalar norm_e = lengths[j];
	const Scalar norm_f = lengths[j + 1];
	const Vec3x& kb = curvatureBinormals[i];

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

	hessTwists[i] = DDtwist;
}

template <int w = BAND_WIDTH>
__global__ void computeTwisting(const int* strand_ptr, Scalar kt, const Scalar* lengths, const Scalar* ilens, const Vec3x* tangents,
	const Vec3x* curvatureBinormals, const Scalar* twists, const Scalar* restTwists,
	Scalar* forces, Scalar* Jacobian)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;
	int start_dof = 4 * strand_ptr[sid] - sid;
	int n_twist = strand_ptr[sid + 1] - strand_ptr[sid] - 2;
	int n_dof = 4 * (n_twist + 2) - 1;
	int global_tid = strand_ptr[sid] - 2 * sid + tid;
	int global_eid = global_tid + sid;
	if (tid >= n_twist) return;

	extern __shared__ Scalar shared_mem[];
	//Vec11x* gradTwists = (Vec11x*)shared_mem;
	Mat11x* hessTwists = (Mat11x*)shared_mem;
	BandMatrixWrapper<w> A(shared_mem + n_twist * 121, n_dof);

	const Scalar ilen = ilens[global_tid];
	const Scalar twist = twists[global_tid];
	const Scalar restTwist = restTwists[global_tid];

	const Vec3x te = tangents[global_eid];
	const Vec3x tf = tangents[global_eid + 1];
	const Scalar norm_e = lengths[global_eid];
	const Scalar norm_f = lengths[global_eid + 1];
	const Vec3x kb = curvatureBinormals[global_tid];

	/* Compute gradTwists */
	Vec11x Dtwist;

	Dtwist.assignAt(0, -0.5 / norm_e * kb);
	Dtwist.assignAt(8, 0.5 / norm_f * kb);
	Dtwist.assignAt(4, -(Dtwist.at<3>(0) + Dtwist.at<3>(8)));
	Dtwist(3) = -1;
	Dtwist(7) = 1;

	/* Compute hessTwists */
	Mat11x& DDtwist = hessTwists[tid];
	DDtwist.setZero();

	Scalar chi = MAX(1e-12, 1 + te.dot(tf));
	const Vec3x tilde_t = 1.0 / chi * (te + tf);

	const Mat3x D2mDe2 = -0.5 / (norm_e * norm_e) * symPart(outerProd(kb, te + tilde_t));
	const Mat3x D2mDf2 = -0.5 / (norm_f * norm_f) * symPart(outerProd(kb, tf + tilde_t));
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

	/* Accumulate to total Jacobian */
	int A_base = start_dof * (2 * w + 1);
	int i, j, idx;
	// global -> shared
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
		for (j = 0; j < 4; ++j) {
			A[idx + j] = Jacobian[A_base + idx + j];
		}
		if (tid == n_twist - 1) {
			for (j = 4; j < 11; ++j) {
				A[idx + j] = Jacobian[A_base + idx + j];
			}
		}
	}
	__syncthreads();

	DDtwist = kt * ilen * (outerProd(Dtwist) + (twist - restTwist) * DDtwist);	// localJ

	// fill in
	int base = 4 * tid;  // local dof idx
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += DDtwist(i, j);
		}
	}
	__syncthreads();
	for (i = 4; i < 8; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += DDtwist(i, j);
		}
	}
	__syncthreads();
	for (i = 8; i < 11; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += DDtwist(i, j);
		}
	}
	__syncthreads();

	// shared -> global
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
		for (j = 0; j < 4; ++j) {
			Jacobian[A_base + idx + j] = A[idx + j];
		}
		if (tid == n_twist - 1) {
			for (j = 4; j < 11; ++j) {
				Jacobian[A_base + idx + j] = A[idx + j];
			}
		}
	}

	/* Accumulate to total force */
	Dtwist = -kt * ilen * (twist - restTwist) * Dtwist;  // localF

	base = start_dof + 4 * tid;

	forces[base] += Dtwist(0);
	forces[base + 1] += Dtwist(1);
	forces[base + 2] += Dtwist(2);
	forces[base + 3] += Dtwist(3);
	__syncthreads();
	forces[base + 4] += Dtwist(4);
	forces[base + 5] += Dtwist(5);
	forces[base + 6] += Dtwist(6);
	forces[base + 7] += Dtwist(7);
	__syncthreads();
	forces[base + 8] += Dtwist(8);
	forces[base + 9] += Dtwist(9);
	forces[base + 10] += Dtwist(10);
}

__global__ void computeTwistingForce(const int* strand_ptr, Scalar kt, const Scalar* twists, const Scalar* restTwists, const Scalar* ilens,
	Scalar* forces, const Vec11x* gradTwists)
{
	int sid = blockIdx.x;
	int i = strand_ptr[sid] - 2 * sid + threadIdx.x;	// global twist idx
	if (i >= strand_ptr[sid + 1] - 2 * sid - 2) return;

	// forces
	Vec11x f = -kt * ilens[i] * (twists[i] - restTwists[i]) * gradTwists[i];
	int base = 4 * i + 7 * sid;

	forces[base] += f(0);
	forces[base + 1] += f(1);
	forces[base + 2] += f(2);
	forces[base + 3] += f(3);
	__syncthreads();
	forces[base + 4] += f(4);
	forces[base + 5] += f(5);
	forces[base + 6] += f(6);
	forces[base + 7] += f(7);
	__syncthreads();
	forces[base + 8] += f(8);
	forces[base + 9] += f(9);
	forces[base + 10] += f(10);
}

template <int w = BAND_WIDTH>
__global__ void computeTwistingGradient(const int* strand_ptr, Scalar kt, const Scalar* twists, const Scalar* restTwists, const Scalar* ilens,
	const Vec11x* gradTwists, const Mat11x* hessTwists, Scalar* Jacobian)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;
	int vid0 = strand_ptr[sid], vid1 = strand_ptr[sid + 1];
	int n_ivtx = vid1 - vid0 - 2;
	if (tid >= n_ivtx) return;

	extern __shared__ Scalar shared_mem[];
	int n_dof = 4 * (vid1 - vid0) - 1;
	int A_base = (4 * vid0 - sid) * (2 * w + 1);
	BandMatrixWrapper<> A(shared_mem, n_dof);
	int i, j, idx;
	// global -> shared
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			A[idx + j] = Jacobian[A_base + idx + j];
		}
		if (tid == n_ivtx - 1) {
#pragma unroll 
			for (j = 4; j < 11; ++j) {
				A[idx + j] = Jacobian[A_base + idx + j];
			}
		}
	}
	__syncthreads();

	// Jocobian
	idx = vid0 - 2 * sid + tid; // global twist idx
	Mat11x localJ = kt * ilens[idx] * (outerProd(gradTwists[idx]) + (twists[idx] - restTwists[idx]) * hessTwists[idx]);

	int base = 4 * tid;  // local dof idx
#pragma unroll
	for (i = 0; i < 4; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
#pragma unroll
	for (i = 4; i < 8; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
#pragma unroll
	for (i = 8; i < 11; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();

	// shared -> global
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			Jacobian[A_base + idx + j] = A[idx + j];
		}
		if (tid == n_ivtx - 1) {
#pragma unroll 
			for (j = 4; j < 11; ++j) {
				Jacobian[A_base + idx + j] = A[idx + j];
			}
		}
	}
}

__global__ void computeTwistingEnergy(int nt, Scalar kt, Scalar* twistingEnergy, const Scalar* twists, const Scalar* restTwists, const Scalar* ilens)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	twistingEnergy[i] = 0.5 * kt * (twists[i] - restTwists[i]) * (twists[i] - restTwists[i]) * ilens[i];
}

__global__ void computeTwistingForce(int nt, const int* ivtx_to_strand, Scalar kt, Scalar* totalForces, const Scalar* twists, const Scalar* restTwists,
	const Scalar* ilens, const Vec11x* gradTwists)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	Vec11x force = -kt * ilens[i] * (twists[i] - restTwists[i]) * gradTwists[i];

	// Add to total forces
	Scalar* start_ptr = totalForces + 4 * i + 7 * ivtx_to_strand[i];
#pragma unroll
	for (int j = 0; j < 11; ++j)
		atomicAdd(start_ptr + j, force(j));
}

__global__ void computeTwistingGradient(int nt, const int* ivtx_to_strand, Scalar kt, MatrixWrapper<Scalar>* gradient, const Scalar* twists, 
	const Scalar* restTwists, const Scalar* ilens, const Vec11x* gradTwists, const Mat11x* hessTwists)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nt) return;

	Mat11x localJ = kt * ilens[i] * (outerProd(gradTwists[i]) + (twists[i] - restTwists[i]) * hessTwists[i]);

	// Add to total hessian
	int idx = 4 * i + 7 * ivtx_to_strand[i];
#pragma unroll
	for (int j = 0; j < 11; ++j)
	{
#pragma unroll
		for (int k = 0; k < 11; ++k) {
			gradient->add(idx + j, idx + k, localJ(j, k));
		}
	}
}

__global__ void computeGradKappa(int nb, const int* ivtx_to_strand, Mat11x4x* gradKappas, const Scalar* lengths, const Vec3x* tangents, 
	const Vec3x* curvatureBinormals,const Vec3x* materialFrames1, const Vec3x* materialFrames2, const Vec4x* kappas)//, Vec11x* grad)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	Mat11x4x gradKappa;

	int j = i + ivtx_to_strand[i];

	const Scalar norm_e = lengths[j];
	const Scalar norm_f = lengths[j + 1];

	const Vec3x& te = tangents[j];
	const Vec3x& tf = tangents[j + 1];

	const Vec3x& m1e = materialFrames1[j];
	const Vec3x& m2e = materialFrames2[j];
	const Vec3x& m1f = materialFrames1[j + 1];
	const Vec3x& m2f = materialFrames2[j + 1];

	Scalar chi = MAX(1e-12, 1.0 + te.dot(tf));

	const Vec3x& tilde_t = (te + tf) / chi;
	const Vec3x& tilde_d1e = (2.0 * m1e) / chi;
	const Vec3x& tilde_d1f = (2.0 * m1f) / chi;
	const Vec3x& tilde_d2e = (2.0 * m2e) / chi;
	const Vec3x& tilde_d2f = (2.0 * m2f) / chi;

	const Vec4x& kappa = kappas[i];

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

	const Vec3x& kb = curvatureBinormals[i];

	gradKappa(3, 0) = -kb.dot(m1e);
	gradKappa(7, 0) = 0.0;
	gradKappa(3, 1) = -kb.dot(m2e);
	gradKappa(7, 1) = 0.0;

	gradKappa(3, 2) = 0.0;
	gradKappa(7, 2) = -kb.dot(m1f);
	gradKappa(3, 3) = 0.0;
	gradKappa(7, 3) = -kb.dot(m2f);

	gradKappas[i] = gradKappa;

	//grad[7 * i] = gradKappa.at<11>(0, 0);
	//grad[7 * i + 1] = gradKappa.at<11>(0, 1);
	//grad[7 * i + 2] = gradKappa.at<11>(0, 2);
	//grad[7 * i + 3] = gradKappa.at<11>(0, 3);

	//grad[7 * i].setZero();
	//grad[7 * i + 1].setZero();
	//grad[7 * i + 2].setZero();
	//grad[7 * i + 3].setZero();

	//grad[7 * i].assignAt(0, tilde_d2e);
	//grad[7 * i].assignAt(4, Dkappa0eDe);
	//grad[7 * i].assignAt(8, Dkappa0eDf);

	//grad[7 * i + 1].assignAt(0, tilde_d1e);
	//grad[7 * i + 1].assignAt(4, Dkappa1eDe);
	//grad[7 * i + 1].assignAt(8, Dkappa1eDf);

	//grad[7 * i + 2].assignAt(0, tilde_d2f);
	//grad[7 * i + 2].assignAt(4, Dkappa0fDe);
	//grad[7 * i + 2].assignAt(8, Dkappa0fDf);

	//grad[7 * i + 3].assignAt(0, tilde_d1f);
	//grad[7 * i + 3].assignAt(4, Dkappa1fDe);
	//grad[7 * i + 3].assignAt(8, Dkappa1fDf);
}

__global__ void computeHessKappa(int nb, const int* ivtx_to_strand, Mat11x* hessKappas, const Scalar* lengths, const Vec3x* tangents, 
	const Vec3x* curvatureBinormals,const Vec3x* materialFrames1, const Vec3x* materialFrames2, const Vec4x* kappas)//, Mat3x* mattest, Vec3x* vectest, const Vec4x* restKappas)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	Mat11x& DDkappa0 = hessKappas[i * 4 + 0];
	Mat11x& DDkappa1 = hessKappas[i * 4 + 1];
	Mat11x& DDkappa2 = hessKappas[i * 4 + 2];
	Mat11x& DDkappa3 = hessKappas[i * 4 + 3];

	DDkappa0.setZero();
	DDkappa1.setZero();
	DDkappa2.setZero();
	DDkappa3.setZero();

	int j = i + ivtx_to_strand[i];

	const Scalar norm_e = lengths[j];
	const Scalar norm_f = lengths[j + 1];

	const Vec3x& te = tangents[j];
	const Vec3x& tf = tangents[j + 1];

	const Vec3x& m1e = materialFrames1[j];
	const Vec3x& m2e = materialFrames2[j];
	const Vec3x& m1f = materialFrames1[j + 1];
	const Vec3x& m2f = materialFrames2[j + 1];

	Scalar chi = MAX(1e-12, 1.0 + te.dot(tf));

	const Vec3x& tilde_t = (te + tf) / chi;
	const Vec3x& tilde_d1e = (2.0 * m1e) / chi;
	const Vec3x& tilde_d2e = (2.0 * m2e) / chi;
	const Vec3x& tilde_d1f = (2.0 * m1f) / chi;
	const Vec3x& tilde_d2f = (2.0 * m2f) / chi;

	const Vec4x& kappa = kappas[i];
	//const Vec4x deltaKappa = kappa - restKappas[i];

	const Vec3x& Dkappa0eDe = 1.0 / norm_e * (-kappa(0) * tilde_t + tf.cross(tilde_d2e));
	const Vec3x& Dkappa0eDf = 1.0 / norm_f * (-kappa(0) * tilde_t - te.cross(tilde_d2e));
	const Vec3x& Dkappa1eDe = 1.0 / norm_e * (-kappa(1) * tilde_t - tf.cross(tilde_d1e));
	const Vec3x& Dkappa1eDf = 1.0 / norm_f * (-kappa(1) * tilde_t + te.cross(tilde_d1e));

	const Vec3x& Dkappa0fDe = 1.0 / norm_e * (-kappa(2) * tilde_t + tf.cross(tilde_d2f));
	const Vec3x& Dkappa0fDf = 1.0 / norm_f * (-kappa(2) * tilde_t - te.cross(tilde_d2f));
	const Vec3x& Dkappa1fDe = 1.0 / norm_e * (-kappa(3) * tilde_t - tf.cross(tilde_d1f));
	const Vec3x& Dkappa1fDf = 1.0 / norm_f * (-kappa(3) * tilde_t + te.cross(tilde_d1f));

	const Vec3x& kb = curvatureBinormals[i];

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

	//mattest[16 * i] = D2kappa0De2 * deltaKappa(0);
	//mattest[16 * i + 1] = D2kappa0Df2 * deltaKappa(0);
	//mattest[16 * i + 2] = D2kappa0DeDf * deltaKappa(0);
	//mattest[16 * i + 3] = D2kappa0DfDe * deltaKappa(0);

	//vectest[16 * i] = D2kappa0DeDthetae * deltaKappa(0);
	//vectest[16 * i + 1] = D2kappa0DeDthetaf * deltaKappa(0);
	//vectest[16 * i + 2] = D2kappa0DfDthetae * deltaKappa(0);
	//vectest[16 * i + 3] = D2kappa0DfDthetaf * deltaKappa(0);

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

	//mattest[16 * i + 4] = D2kappa1De2 * deltaKappa(1);
	//mattest[16 * i + 5] = D2kappa1Df2 * deltaKappa(1);
	//mattest[16 * i + 6] = D2kappa1DeDf * deltaKappa(1);
	//mattest[16 * i + 7] = D2kappa1DfDe * deltaKappa(1);

	//vectest[16 * i + 4] = D2kappa1DeDthetae * deltaKappa(1);
	//vectest[16 * i + 5] = D2kappa1DeDthetaf * deltaKappa(1);
	//vectest[16 * i + 6] = D2kappa1DfDthetae * deltaKappa(1);
	//vectest[16 * i + 7] = D2kappa1DfDthetaf * deltaKappa(1);

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

	//mattest[16 * i + 8] = D2kappa2De2 * deltaKappa(2);
	//mattest[16 * i + 9] = D2kappa2Df2 * deltaKappa(2);
	//mattest[16 * i + 10] = D2kappa2DeDf * deltaKappa(2);
	//mattest[16 * i + 11] = D2kappa2DfDe * deltaKappa(2);

	//vectest[16 * i + 8] = D2kappa2DeDthetae * deltaKappa(2);
	//vectest[16 * i + 9] = D2kappa2DeDthetaf * deltaKappa(2);
	//vectest[16 * i + 10] = D2kappa2DfDthetae * deltaKappa(2);
	//vectest[16 * i + 11] = D2kappa2DfDthetaf * deltaKappa(2);

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

	//mattest[16 * i + 12] = D2kappa3De2 * deltaKappa(3);
	//mattest[16 * i + 13] = D2kappa3Df2 * deltaKappa(3);
	//mattest[16 * i + 14] = D2kappa3DeDf * deltaKappa(3);
	//mattest[16 * i + 15] = D2kappa3DfDe * deltaKappa(3);

	//vectest[16 * i + 12] = D2kappa3DeDthetae * deltaKappa(3);
	//vectest[16 * i + 13] = D2kappa3DeDthetaf * deltaKappa(3);
	//vectest[16 * i + 14] = D2kappa3DfDthetae * deltaKappa(3);
	//vectest[16 * i + 15] = D2kappa3DfDthetaf * deltaKappa(3);

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


__global__ void computeBendingEnergy(int nb, Scalar kb, Scalar* bendingEnergy, const Vec4x* kappas, const Vec4x* restKappas, const Scalar* ilens)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	extern __shared__ Scalar energy[];
	bendingEnergy[i] = 0.25 * kb * ilens[i] * (kappas[i] - restKappas[i]).norm2();
}

template <int w = BAND_WIDTH>
__global__ void computeBending(const int* strand_ptr, Scalar Kb, const Scalar* lengths, const Scalar* ilens, 
	const Vec3x* tangents, const Vec3x* materialFrames1, const Vec3x* materialFrames2,
	const Vec3x* curvatureBinormals, const Vec4x* kappas, const Vec4x* restKappas,
	Scalar* forces, Scalar* Jacobian)//, Mat3x* mattest, Vec3x* vectest, Vec11x* gradtest, Mat11x* localJtest,  bool az)
{
	int sid = blockIdx.x;
	int bid = threadIdx.x;
	int start_dof = 4 * strand_ptr[sid] - sid;
	int n_bend = strand_ptr[sid + 1] - strand_ptr[sid] - 2;
	int n_dof = 4 * (n_bend + 2) - 1;
	int global_bid = strand_ptr[sid] - 2 * sid + bid;
	int global_eid = global_bid + sid;
	if (bid >= n_bend) return;

	extern __shared__ Scalar shared_mem[];
	Mat11x* J = (Mat11x*)shared_mem;
	Vec11x* F = (Vec11x*)(shared_mem + 11 * 11 * n_bend);

	const Scalar ilen = ilens[global_bid];
	const Vec4x kappa = kappas[global_bid];
	const Vec4x deltaKappa = kappa - restKappas[global_bid];

	const Scalar norm_e = lengths[global_eid];
	const Scalar norm_f = lengths[global_eid + 1];

	const Vec3x te = tangents[global_eid];
	const Vec3x tf = tangents[global_eid + 1];

	const Vec3x m1e = materialFrames1[global_eid];
	const Vec3x m2e = materialFrames2[global_eid];
	const Vec3x m1f = materialFrames1[global_eid + 1];
	const Vec3x m2f = materialFrames2[global_eid + 1];

	const Vec3x kb = curvatureBinormals[global_bid];

	Vec11x& localF = F[bid];
	Mat11x& localJ = J[bid];
	localF.setZero(); localJ.setZero();

	int i, j, idx;

	/* Accumulate gradKappa and hessKappa */

	Scalar chi = MAX(1e-12, 1.0 + te.dot(tf));

	const Vec3x tilde_t = (te + tf) / chi;

	const Mat3x Id = Mat3x::Identity();

	const Vec3x DchiDe = 1.0 / norm_e * (Id - outerProd(te, te)) * tf;
	const Vec3x DchiDf = 1.0 / norm_f * (Id - outerProd(tf, tf)) * te;

	const Mat3x DtfDf = 1.0 / norm_f * (Id - outerProd(tf, tf));
	const Mat3x DttDe = 1.0 / (chi * norm_e) * ((Id - outerProd(te, te)) - outerProd(tilde_t, (Id - outerProd(te, te)) * tf));
	const Mat3x DttDf = 1.0 / (chi * norm_f) * ((Id - outerProd(tf, tf)) - outerProd(tilde_t, (Id - outerProd(tf, tf)) * te));

	Scalar sign;
	Vec3x m1, m2, tilde_m;

	Vec3x DkappaDe, DkappaDf;
	Vec11x gradKappa;

	Mat3x D2kappaDe2, D2kappaDf2, D2kappaDeDf, D2kappaDfDe;
	Scalar D2kappaDthetae2, D2kappaDthetaf2;
	Vec3x D2kappaDeDthetae, D2kappaDeDthetaf, D2kappaDfDthetae, D2kappaDfDthetaf;

	for (i = 0; i < 4; ++i)
	{
		switch (i)
		{
		case 0: sign = 1.0; m1 = m2e; m2 = m1e; break;
		case 1: sign = -1.0; m1 = m1e; m2 = m2e; break;
		case 2: sign = 1.0; m1 = m2f; m2 = m1f; break;
		case 3: sign = -1.0; m1 = m1f; m2 = m2f; break;
		}

		tilde_m = (2.0 * m1) / chi;

		// gradKappa
		DkappaDe = 1.0 / norm_e * (-kappa(i) * tilde_t + sign * tf.cross(tilde_m));
		DkappaDf = 1.0 / norm_f * (-kappa(i) * tilde_t - sign * te.cross(tilde_m));

		gradKappa.assignAt(0, -DkappaDe);
		gradKappa.assignAt(4, DkappaDe - DkappaDf);
		gradKappa.assignAt(8, DkappaDf);
		if (i < 2) {
			gradKappa(3) = -kb.dot(m2);
			gradKappa(7) = 0.0;
		}
		else {
			gradKappa(3) = 0.0;
			gradKappa(7) = -kb.dot(m2);
		}

		localF += deltaKappa(i) * gradKappa;
		localJ += outerProd(gradKappa);

		//gradtest[7 * global_bid + i] = gradKappa;
		//gradtest[7 * global_bid + i].setZero();
		//gradtest[7 * global_bid + i].assignAt(0, tilde_m);
		//gradtest[7 * global_bid + i].assignAt(4, DkappaDe);
		//gradtest[7 * global_bid + i].assignAt(8, DkappaDf);

		// hessKappa
		D2kappaDe2 = -deltaKappa(i) / norm_e * symPart(outerProd(tilde_t + te, DkappaDe) + kappa(i) * DttDe + outerProd(1.0 / chi * tf.cross(tilde_m), DchiDe));
		D2kappaDf2 = -deltaKappa(i) / norm_f * symPart(outerProd(tilde_t + tf, DkappaDf) + kappa(i) * DttDf + outerProd(1.0 / chi * te.cross(tilde_m), DchiDf));
		D2kappaDeDf = -deltaKappa(i) / norm_e * (outerProd(tilde_t, DkappaDf) + kappa(i) * DttDf + sign * outerProd(1.0 / chi * tf.cross(tilde_m), DchiDf) + sign * crossMat(tilde_m) * DtfDf);
		D2kappaDfDe = D2kappaDeDf.transpose();
		if (i < 2) {
			D2kappaDthetae2 = -sign * kb.dot(m1) * deltaKappa(i);
			D2kappaDthetaf2 = 0.0;
			D2kappaDeDthetae = deltaKappa(i) / norm_e * (kb.dot(m2) * tilde_t - 2.0 / chi * tf.cross(m2));
			D2kappaDeDthetaf = Vec3x::Zero();
			D2kappaDfDthetae = deltaKappa(i) / norm_f * (kb.dot(m2) * tilde_t + 2.0 / chi * te.cross(m2));
			D2kappaDfDthetaf = Vec3x::Zero();
		}
		else {
			D2kappaDthetae2 = 0.0;
			D2kappaDthetaf2 = -sign * kb.dot(m1) * deltaKappa(i);
			D2kappaDeDthetae = Vec3x::Zero();
			D2kappaDeDthetaf = deltaKappa(i) / norm_e * (kb.dot(m2) * tilde_t - 2.0 / chi * tf.cross(m2));
			D2kappaDfDthetae = Vec3x::Zero();
			D2kappaDfDthetaf = deltaKappa(i) / norm_f * (kb.dot(m2) * tilde_t + 2.0 / chi * te.cross(m2));
		}

		//mattest[16 * global_bid + 4 * i] = D2kappaDe2;
		//mattest[16 * global_bid + 4 * i + 1] = D2kappaDf2;
		//mattest[16 * global_bid + 4 * i + 2] = D2kappaDeDf;
		//mattest[16 * global_bid + 4 * i + 3] = D2kappaDfDe;

		//vectest[16 * global_bid + 4 * i] = D2kappaDeDthetae;
		//vectest[16 * global_bid + 4 * i + 1] = D2kappaDeDthetaf;
		//vectest[16 * global_bid + 4 * i + 2] = D2kappaDfDthetae;
		//vectest[16 * global_bid + 4 * i + 3] = D2kappaDfDthetaf;

		localJ.addAt(0, 0, D2kappaDe2);
		localJ.addAt(0, 4, -D2kappaDe2 + D2kappaDeDf);
		localJ.addAt(4, 0, -D2kappaDe2 + D2kappaDfDe);
		localJ.addAt(4, 4, D2kappaDe2 - (D2kappaDeDf + D2kappaDfDe) + D2kappaDf2);
		localJ.addAt(0, 8, -D2kappaDeDf);
		localJ.addAt(8, 0, -D2kappaDfDe);
		localJ.addAt(4, 8, D2kappaDeDf - D2kappaDf2);
		localJ.addAt(8, 4, D2kappaDfDe - D2kappaDf2);
		localJ.addAt(8, 8, D2kappaDf2);
		localJ(3, 3) += D2kappaDthetae2;
		localJ(7, 7) += D2kappaDthetaf2;
		localJ.addAt<true>(0, 3, -D2kappaDeDthetae);
		localJ.addAt<false>(3, 0, -D2kappaDeDthetae);
		localJ.addAt<true>(4, 3, D2kappaDeDthetae - D2kappaDfDthetae);
		localJ.addAt<false>(3, 4, D2kappaDeDthetae - D2kappaDfDthetae);
		localJ.addAt<true>(8, 3, D2kappaDfDthetae);
		localJ.addAt<false>(3, 8, D2kappaDfDthetae);
		localJ.addAt<true>(0, 7, -D2kappaDeDthetaf);
		localJ.addAt<false>(7, 0, -D2kappaDeDthetaf);
		localJ.addAt<true>(4, 7, D2kappaDeDthetaf - D2kappaDfDthetaf);
		localJ.addAt<false>(7, 4, D2kappaDeDthetaf - D2kappaDfDthetaf);
		localJ.addAt<true>(8, 7, D2kappaDfDthetaf);
		localJ.addAt<false>(7, 8, D2kappaDfDthetaf);
	}

	/* Add localF to total forces */
	//gradtest[7 * global_bid + 4] = localF;
	localF *= -0.5 * Kb * ilen;
	//gradtest[7 * global_bid + 5] = localF;
	//gradtest[7 * global_bid + 6] = localF;

	int base = start_dof + 4 * bid;

	forces[base] += localF(0);
	forces[base + 1] += localF(1);
	forces[base + 2] += localF(2);
	forces[base + 3] += localF(3);
	__syncthreads();
	forces[base + 4] += localF(4);
	forces[base + 5] += localF(5);
	forces[base + 6] += localF(6);
	forces[base + 7] += localF(7);
	__syncthreads();
	forces[base + 8] += localF(8);
	forces[base + 9] += localF(9);
	forces[base + 10] += localF(10);
	__syncthreads();

	/* Add localJ to total Jacobian */
	BandMatrixWrapper<w> A(shared_mem + 11 * 11 * n_bend, n_dof);	// reuse F's memory
	// global -> shared
	base = start_dof * (2 * w + 1);
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * bid;
		for (j = 0; j < 4; ++j) {
			A[idx + j] = Jacobian[base + idx + j];
		}
		if (bid == n_bend - 1) {
			for (j = 4; j < 11; ++j) {
				A[idx + j] = Jacobian[base + idx + j];
			}
		}
	}
	__syncthreads();

	//localJtest[global_bid] = localJ;
	localJ *= 0.5 * Kb * ilen;

	// fill in
	base = 4 * bid;
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
	for (i = 4; i < 8; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
	for (i = 8; i < 11; ++i) {
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();

	// shared -> global
	base = start_dof * (2 * w + 1);
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * bid;
		for (j = 0; j < 4; ++j) {
			Jacobian[base + idx + j] = A[idx + j];
		}
		if (bid == n_bend - 1) {
			for (j = 4; j < 11; ++j) {
				Jacobian[base + idx + j] = A[idx + j];
			}
		}
	}
}

__global__ void computeBendingForce(int nb, const int* ivtx_to_strand, Scalar kb, Scalar* totalForces, const Vec4x* kappas, const Vec4x* restKappas,
	const Scalar* ilens, const Mat11x4x* gradKappas)//, Vec11x* grad)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	Vec11x force = -0.5 * kb * ilens[i] * gradKappas[i] * (kappas[i] - restKappas[i]);
	//grad[7 * i + 4] = gradKappas[i] * (kappas[i] - restKappas[i]);
	//grad[7 * i + 5] = -0.5 * kb * ilens[i] * (gradKappas[i] * (kappas[i] - restKappas[i]));
	//grad[7 * i + 6] = force;

	Scalar* start_ptr = totalForces + 4 * i + 7 * ivtx_to_strand[i];
#pragma unroll
	for (int j = 0; j < 11; ++j)
		atomicAdd(start_ptr + j, force(j));
}

template <int w = BAND_WIDTH>
__global__ void computeBendingForce(const int* strand_ptr, Scalar kb, Scalar* forces, const Vec4x* kappas, const Vec4x* restKappas,
	const Scalar* ilens, const Mat11x4x* gradKappas)
{
	int sid = blockIdx.x;
	int i = strand_ptr[sid] - 2 * sid + threadIdx.x;	// global bend idx
	if (i >= strand_ptr[sid + 1] - 2 * sid - 2) return;

	// forces
	Vec11x f = -0.5 * kb * ilens[i] * gradKappas[i] * (kappas[i] - restKappas[i]);

	int base = 4 * i + 7 * sid;

	forces[base] += f(0);
	forces[base + 1] += f(1);
	forces[base + 2] += f(2);
	forces[base + 3] += f(3);
	__syncthreads();
	forces[base + 4] += f(4);
	forces[base + 5] += f(5);
	forces[base + 6] += f(6);
	forces[base + 7] += f(7);
	__syncthreads();
	forces[base + 8] += f(8);
	forces[base + 9] += f(9);
	forces[base + 10] += f(10);
}

__global__ void computeBendingGradient(int nb, const int* ivtx_to_strand, Scalar kb, MatrixWrapper<Scalar>* gradient, const Vec4x* kappas,
	const Vec4x* restKappas, const Scalar* ilens, const Mat11x4x* gradKappas, const Mat11x* hessKappas)//, bool az, Mat11x* localJtest)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= nb) return;

	Mat11x localJ = symProd(gradKappas[i]);

	Vec4x deltaKappa = kappas[i] - restKappas[i];
	for (int idx = 0; idx < 4; ++idx) {
		localJ += hessKappas[4 * i + idx] * deltaKappa(idx);
	}
	//localJtest[i] = localJ;
	localJ *= 0.5 * kb * ilens[i];

	int start_idx = 4 * i + 7 * ivtx_to_strand[i];
#pragma unroll
	for (int j = 0; j < 11; ++j)
	{
#pragma unroll
		for (int k = 0; k < 11; ++k) {
			gradient->add(start_idx + j, start_idx + k, localJ(j, k));
		}
	}
}

template <int w = BAND_WIDTH>
__global__ void computeBendingGradient(const int* strand_ptr, Scalar kb, const Vec4x* kappas, const Vec4x* restKappas,
	const Scalar* ilens, const Mat11x4x* gradKappas, const Mat11x* hessKappas, Scalar* Jacobian)
{
	int sid = blockIdx.x;
	int tid = threadIdx.x;
	int vid0 = strand_ptr[sid], vid1 = strand_ptr[sid + 1];
	int n_ivtx = vid1 - vid0 - 2;
	if (tid >= n_ivtx) return;

	extern __shared__ Scalar shared_mem[];
	int n_dof = 4 * (vid1 - vid0) - 1;
	int A_base = (4 * vid0 - sid) * (2 * w + 1);
	BandMatrixWrapper<> A(shared_mem, n_dof);
	int i, j, idx;
	// global -> shared
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			A[idx + j] = Jacobian[A_base + idx + j];
		}
		if (tid == n_ivtx - 1) {
#pragma unroll 
			for (j = 4; j < 11; ++j) {
				A[idx + j] = Jacobian[A_base + idx + j];
			}
		}
	}
	__syncthreads();

	// Jocobian
	idx = vid0 - 2 * sid + tid; // global twist idx
	Mat11x localJ = symProd(gradKappas[idx]);
	Vec4x deltaKappa = kappas[idx] - restKappas[idx];
#pragma unroll
	for (i = 0; i < 4; ++i)
		localJ += hessKappas[4 * idx + i] * deltaKappa(i);
	localJ *= 0.5 * kb * ilens[idx];

	int base = 4 * tid;  // local dof idx
#pragma unroll
	for (i = 0; i < 4; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
#pragma unroll
	for (i = 4; i < 8; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();
#pragma unroll
	for (i = 8; i < 11; ++i) {
#pragma unroll
		for (j = 0; j < 11; ++j) {
			A(base + i, base + j) += localJ(i, j);
		}
	}
	__syncthreads();

	// shared -> global
#pragma unroll
	for (i = 0; i < 2 * w + 1; ++i) {
		idx = i * n_dof + 4 * tid;
#pragma unroll
		for (j = 0; j < 4; ++j) {
			Jacobian[A_base + idx + j] = A[idx + j];
		}
		if (tid == n_ivtx - 1) {
#pragma unroll 
			for (j = 4; j < 11; ++j) {
				Jacobian[A_base + idx + j] = A[idx + j];
			}
		}
	}
}

StrandStates::StrandStates(const StrandParameters* params, int num_dofs, int num_strands, int num_vertices, int max_vtx,
	int num_fixed, int* fixed_idx, const int* strand_ptr, const int* vtx_to_strand, const int* edge_to_strand, 
	const int* inner_vtx_to_strand, Vec3x* fixed_targets, HessianMatrix* hessian) :
	m_params(params),
	m_numDofs(num_dofs),
	m_numStrands(num_strands),
	m_numVertices(num_vertices),
	m_numEdges(num_vertices - num_strands),
	m_numInnerVtx(num_vertices - 2 * num_strands),
	m_numFixed(num_fixed),
	m_maxVtxPerStrand(max_vtx),
	m_fixed_idx(fixed_idx),
	m_strand_ptr(strand_ptr),
	m_vertex_to_strand(vtx_to_strand),
	m_edge_to_strand(edge_to_strand),
	m_inner_vtx_to_strand(inner_vtx_to_strand),
	m_fixedTargets(fixed_targets),
	m_totalJacobian(hessian)
{
	checkCudaErrors(cublasCreate(&m_cublasHandle));

	// Allocates global memory
	checkCudaErrors(cudaMalloc((void**)& m_mass, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_gravity, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_invVtxLengths, m_numInnerVtx * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_restLengths, m_numEdges * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_restTwists, m_numInnerVtx * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_restKappas, m_numInnerVtx * sizeof(Vec4x)));

	checkCudaErrors(cudaMalloc((void**)& m_x, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_lengths, m_numEdges * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_tangents, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_referenceFrames1, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_twists, m_numInnerVtx * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_materialFrames1, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_materialFrames2, m_numEdges * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_curvatureBinormals, m_numInnerVtx * sizeof(Vec3x)));
	checkCudaErrors(cudaMalloc((void**)& m_kappas, m_numInnerVtx * sizeof(Vec4x)));

	checkCudaErrors(cudaMalloc((void**)& m_fixingEnergy, m_numFixed * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_stretchingEnergy, m_numEdges * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_twistingEnergy, m_numInnerVtx * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_bendingEnergy, m_numInnerVtx * sizeof(Scalar)));

	checkCudaErrors(cudaMalloc((void**)& m_totalForces, m_numDofs * sizeof(Scalar)));
	checkCudaErrors(cudaMalloc((void**)& m_gradTwists, m_numInnerVtx * sizeof(Vec11x)));
	checkCudaErrors(cudaMalloc((void**)& m_gradKappas, m_numInnerVtx * sizeof(Mat11x4x)));

	checkCudaErrors(cudaMalloc((void**)& m_hessTwists, m_numInnerVtx * sizeof(Mat11x)));
	checkCudaErrors(cudaMalloc((void**)& m_hessKappas, 4 * m_numInnerVtx * sizeof(Mat11x)));

	strand_blocksPerGrid = (m_numStrands + g_threadsPerBlock - 1) / g_threadsPerBlock;
	vtx_blocksPerGrid = (m_numVertices + g_threadsPerBlock - 1) / g_threadsPerBlock;
	edge_blocksPerGrid = (m_numEdges + g_threadsPerBlock - 1) / g_threadsPerBlock;
	ivtx_blocksPerGrid = (m_numInnerVtx + g_threadsPerBlock - 1) / g_threadsPerBlock;
	fixed_blocksPerGrid = (m_numFixed + g_threadsPerBlock - 1) / g_threadsPerBlock;
}

StrandStates::~StrandStates()
{
	cublasDestroy(m_cublasHandle);

	cudaFree(m_mass);
	cudaFree(m_gravity);
	cudaFree(m_invVtxLengths);
	cudaFree(m_restLengths);
	cudaFree(m_restTwists);
	cudaFree(m_restKappas);

	cudaFree(m_x);
	cudaFree(m_lengths);
	cudaFree(m_tangents);
	cudaFree(m_referenceFrames1);
	cudaFree(m_twists);
	cudaFree(m_materialFrames1);
	cudaFree(m_materialFrames2);
	cudaFree(m_curvatureBinormals);
	cudaFree(m_kappas);

	cudaFree(m_fixingEnergy);
	cudaFree(m_stretchingEnergy);
	cudaFree(m_twistingEnergy);
	cudaFree(m_bendingEnergy);

	cudaFree(m_totalForces);
	cudaFree(m_gradTwists);
	cudaFree(m_gradKappas);
	cudaFree(m_hessTwists);
	cudaFree(m_hessKappas);
}

void StrandStates::init()
{
	updateEdges <<< edge_blocksPerGrid, g_threadsPerBlock >>> (m_numEdges, m_edge_to_strand, m_x, m_restLengths, m_tangents);

	Scalar mass_per_len = M_PI * m_params->m_radius * m_params->m_radius * m_params->m_density;

	initMass <<< vtx_blocksPerGrid, g_threadsPerBlock >>> (
		m_numVertices, m_strand_ptr, m_vertex_to_strand, m_restLengths, mass_per_len, m_params->m_radius, m_mass);

	cudaMemset(m_gravity, 0, m_numDofs * sizeof(Scalar));
	initGravity <<< vtx_blocksPerGrid, g_threadsPerBlock >>> (m_numVertices, m_vertex_to_strand, m_mass, -981.0, m_gravity);

	initInvVtxLengths <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (m_numInnerVtx, m_inner_vtx_to_strand, m_restLengths, m_invVtxLengths);
	initRefFrames <<< strand_blocksPerGrid, g_threadsPerBlock >>> (m_numStrands, m_strand_ptr, m_tangents, m_referenceFrames1);
	initMatFrames <<< edge_blocksPerGrid, g_threadsPerBlock >>> (
		m_numEdges, m_edge_to_strand, m_x, m_tangents, m_referenceFrames1, m_materialFrames1, m_materialFrames2);

	updateTwists <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
		m_numInnerVtx, m_inner_vtx_to_strand, m_x, m_tangents, m_referenceFrames1, m_restTwists);
	updateKappas <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
		m_numInnerVtx, m_inner_vtx_to_strand, m_x, m_tangents, m_materialFrames1, m_materialFrames2, m_curvatureBinormals, m_restKappas);

	//Eigen::VecXx test(3 * m_numEdges);
	//cudaMemcpy(test.data(), m_referenceFrames1, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < m_numEdges; ++i) {
	//	printf("{%.10f, %.10f, %.10f}, ", test(3 * i), test(3 * i + 1), test(3 * i + 2));
	//}
	//printf("\n");
	//cudaMemcpy(test.data(), m_materialFrames1, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < m_numEdges; ++i) {
	//	printf("{%.10f, %.10f, %.10f}, ", test(3 * i), test(3 * i + 1), test(3 * i + 2));
	//}
	//printf("\n");
}

void StrandStates::updateStates(const Vec3x* last_tangents, const Vec3x* last_ref1)
{
	// Updates position depandent variables
	updateEdges << < edge_blocksPerGrid, g_threadsPerBlock >> > (m_numEdges, m_edge_to_strand, m_x, m_lengths, m_tangents);
	updateFrames << < edge_blocksPerGrid, g_threadsPerBlock >> > (
		m_numEdges, m_edge_to_strand, m_x, last_ref1, last_tangents, m_tangents, m_referenceFrames1, m_materialFrames1, m_materialFrames2);
	updateTwists << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
		m_numInnerVtx, m_inner_vtx_to_strand, m_x, m_tangents, m_referenceFrames1, m_twists);
	updateKappas << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
		m_numInnerVtx, m_inner_vtx_to_strand, m_x, m_tangents, m_materialFrames1, m_materialFrames2, m_curvatureBinormals, m_kappas);
}

void StrandStates::computeForcesAndJacobian(bool az, bool withStretch, bool withTwist, bool withBend)
{
	m_timing.reset();
	cudaMemset(m_totalForces, 0, m_numDofs * sizeof(Scalar));
	//m_totalJacobian->setZero();
	//Mat3x* d_mats;
	//cudaMalloc((void**)&d_mats, m_numInnerVtx * 16 * sizeof(Mat3x));
	//Vec3x* d_vec;
	//cudaMalloc((void**)&d_vec, m_numInnerVtx * 16 * sizeof(Vec3x));
	//Vec11x* d_grad;
	//cudaMalloc((void**)&d_grad, m_numInnerVtx * 7 * sizeof(Vec11x));
	//Mat11x* d_localJ;
	//cudaMalloc((void**)&d_localJ, m_numInnerVtx * sizeof(Mat11x));

	if (az)
	{

		// Stretching
		if (withStretch) 
		{
			// Fixing
			m_timer.start();
			computeFixingForces << < fixed_blocksPerGrid, g_threadsPerBlock >> > (m_numFixed, m_params->m_ks, m_fixed_idx, m_vertex_to_strand, m_fixedTargets, m_x, m_totalForces);
			m_timing.fixingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			//computeFixingGradient <<< fixed_blocksPerGrid, g_threadsPerBlock >>> (
			//	m_numFixed, m_params->m_ks, m_fixed_idx, m_vertex_to_strand, m_totalJacobian->getMatrix());
			computeFixingGradient << < fixed_blocksPerGrid, g_threadsPerBlock >> > (10, m_numFixed, m_params->m_ks, m_strand_ptr, m_fixed_idx, m_vertex_to_strand, m_totalJacobian->getBandValues());
			m_timing.fixingHessian = m_timer.elapsedMilliseconds();

			m_timer.start();
			//computeStretchingForces <<< edge_blocksPerGrid, g_threadsPerBlock >>> (
			//	m_numEdges, m_params->m_ks, m_edge_to_strand, m_totalForces, m_lengths, m_restLengths, m_tangents);
			computeStretchingForces << < m_numStrands, threadSizeof(g_maxVertex) >> > (m_strand_ptr, m_params->m_ks, m_lengths, m_restLengths, m_tangents, m_totalForces);
			m_timing.stretchingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			//computeStretchingGradient <<< edge_blocksPerGrid, g_threadsPerBlock >>> (
			//	m_numEdges, m_params->m_ks, m_edge_to_strand, m_totalJacobian->getMatrix(), m_lengths, m_restLengths, m_tangents);
			computeStretchingGradient<> << < m_numStrands, threadSizeof(g_maxVertex), 21 * (4 * g_maxVertex - 1) * sizeof(Scalar) >> > (m_strand_ptr, m_params->m_ks, m_totalJacobian->getBandValues(), m_lengths, m_restLengths, m_tangents);
			m_timing.stretchingHessian = m_timer.elapsedMilliseconds();
		}

		int numbytes = 65536;  // 64KB
		cudaFuncSetAttribute(computeTwisting<>, cudaFuncAttributeMaxDynamicSharedMemorySize, numbytes);
		cudaFuncSetAttribute(computeBending<>, cudaFuncAttributeMaxDynamicSharedMemorySize, numbytes);
		numbytes = (11 * 11 * (g_maxVertex - 2) + (4 * g_maxVertex - 1) * (2 * BAND_WIDTH + 1)) * sizeof(Scalar);

		// Twisting 
		if (withTwist)
		{
			m_timer.start();
			//computeGradTwist << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
			//	m_numInnerVtx, m_inner_vtx_to_strand, m_gradTwists, m_curvatureBinormals, m_lengths);
			////computeTwistingForce <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
			////	m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kt, m_totalForces, m_twists, m_restTwists, m_invVtxLengths, m_gradTwists);
			//computeTwistingForce << < m_numStrands, 2 * g_threadsPerBlock >> > (m_strand_ptr, m_params->m_kt, m_twists, m_restTwists, m_invVtxLengths, m_totalForces, m_gradTwists);
			//m_timing.twistingGradient = m_timer.elapsedMilliseconds();
			//m_timer.start();
			//computeHessTwist << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
			//	m_numInnerVtx, m_inner_vtx_to_strand, m_hessTwists, m_tangents, m_lengths, m_curvatureBinormals);
			////computeTwistingGradient <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
			////	m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kt, m_totalJacobian->getMatrix(), m_twists, m_restTwists, m_invVtxLengths, m_gradTwists, m_hessTwists);
			//computeTwistingGradient<> << < m_numStrands, 2 * g_threadsPerBlock, 21 * (4 * g_maxVertex - 1) * sizeof(Scalar) >> >
			//	(m_strand_ptr, m_params->m_kt, m_twists, m_restTwists, m_invVtxLengths, m_gradTwists, m_hessTwists, m_totalJacobian->getBandValues());
			computeTwisting<> << < m_numStrands, threadSizeof(g_maxVertex - 2), numbytes >> >
				(m_strand_ptr, m_params->m_kt, m_lengths, m_invVtxLengths, m_tangents, m_curvatureBinormals, m_twists, m_restTwists, m_totalForces, m_totalJacobian->getBandValues());
			m_timing.twistingHessian = m_timer.elapsedMilliseconds();
		}

		// Bending
		m_timer.start();
		if (withBend)
		{
			//computeGradKappa << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
			//	m_numInnerVtx, m_inner_vtx_to_strand, m_gradKappas, m_lengths, m_tangents, m_curvatureBinormals, m_materialFrames1, m_materialFrames2, m_kappas);
			////computeBendingForce <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
			////	m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kb, m_totalForces, m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas);
			//computeBendingForce << < m_numStrands, 2 * g_threadsPerBlock >> > (m_strand_ptr, m_params->m_kb, m_totalForces, m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas);
			//m_timing.bendingGradient = m_timer.elapsedMilliseconds();
			//m_timer.start();
			//computeHessKappa << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
			//	m_numInnerVtx, m_inner_vtx_to_strand, m_hessKappas, m_lengths, m_tangents, m_curvatureBinormals, m_materialFrames1, m_materialFrames2, m_kappas);
			////computeBendingGradient <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (
			////	m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kb, m_totalJacobian->getMatrix(), m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas, m_hessKappas);
			//computeBendingGradient<> << < m_numStrands, 2 * g_threadsPerBlock, 21 * (4 * g_maxVertex - 1) * sizeof(Scalar) >> >
			//	(m_strand_ptr, m_params->m_kb, m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas, m_hessKappas, m_totalJacobian->getBandValues());
			computeBending<> << < m_numStrands, threadSizeof(g_maxVertex - 2), numbytes >> >
				(m_strand_ptr, m_params->m_kb, m_lengths, m_invVtxLengths, m_tangents, m_materialFrames1, m_materialFrames2, m_curvatureBinormals, m_kappas, m_restKappas, m_totalForces, m_totalJacobian->getBandValues());// , d_mats, d_vec, d_grad, d_localJ, true);
			m_timing.bendingHessian = m_timer.elapsedMilliseconds();
		}
	}
	else
	{

		// Stretching
		if (withStretch)
		{
			// Fixing
			m_timer.start();
			computeFixingForces << < fixed_blocksPerGrid, g_threadsPerBlock >> > (m_numFixed, m_params->m_ks, m_fixed_idx, m_vertex_to_strand, m_fixedTargets, m_x, m_totalForces);
			m_timing.fixingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			computeFixingGradient << < fixed_blocksPerGrid, g_threadsPerBlock >> > (
				m_numFixed, m_params->m_ks, m_fixed_idx, m_vertex_to_strand, m_totalJacobian->getMatrix());
			m_timing.fixingHessian = m_timer.elapsedMilliseconds();

			m_timer.start();
			computeStretchingForces << < edge_blocksPerGrid, g_threadsPerBlock >> > (
				m_numEdges, m_params->m_ks, m_edge_to_strand, m_totalForces, m_lengths, m_restLengths, m_tangents);
			m_timing.stretchingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			computeStretchingGradient << < edge_blocksPerGrid, g_threadsPerBlock >> > (
				m_numEdges, m_params->m_ks, m_edge_to_strand, m_totalJacobian->getMatrix(), m_lengths, m_restLengths, m_tangents);
			m_timing.stretchingHessian = m_timer.elapsedMilliseconds();
		}

		// Twisting 
		if (withTwist)
		{
			m_timer.start();
			computeGradTwist << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_gradTwists, m_curvatureBinormals, m_lengths);
			computeTwistingForce << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kt, m_totalForces, m_twists, m_restTwists, m_invVtxLengths, m_gradTwists);
			m_timing.twistingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			computeHessTwist << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_hessTwists, m_tangents, m_lengths, m_curvatureBinormals);
			computeTwistingGradient << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kt, m_totalJacobian->getMatrix(), m_twists, m_restTwists, m_invVtxLengths, m_gradTwists, m_hessTwists);
			m_timing.twistingHessian = m_timer.elapsedMilliseconds();
		}

		//Eigen::VecXx test(3 * m_numEdges);
		//cudaMemcpy(test.data(), m_referenceFrames1, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < m_numEdges; ++i) {
		//	printf("{%.10f, %.10f, %.10f}, ", test(3 * i), test(3 * i + 1), test(3 * i + 2));
		//}
		//printf("\n");
		//cudaMemcpy(test.data(), m_materialFrames1, test.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < m_numEdges; ++i) {
		//	printf("{%.10f, %.10f, %.10f}, ", test(3 * i), test(3 * i + 1), test(3 * i + 2));
		//}
		//printf("\n");

		// Bending
		if (withBend)
		{
			//m_timer.start();
			//Eigen::VecXx gradKappa(m_numInnerVtx * 3 * 13);
			//Vec3x* d_gradKappa;
			//cudaMalloc((void**)&d_gradKappa, gradKappa.size() * sizeof(Scalar));
			computeGradKappa << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_gradKappas, m_lengths, m_tangents, m_curvatureBinormals, m_materialFrames1, m_materialFrames2, m_kappas);// , d_grad);
			//cudaMemcpy(gradKappa.data(), d_gradKappa, gradKappa.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
			//for (int i = 0; i < m_numInnerVtx; ++i) {
			//	for (int j = 0; j < 13; ++j)
			//		printf("{%.10f, %.10f, %.10f}, ", gradKappa(39 * i + 3 * j), gradKappa(39 * i + 3 * j + 1), gradKappa(39 * i + 3 * j + 2));
			//	printf("\n");
			//}
			computeBendingForce << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kb, m_totalForces, m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas);//, d_grad);
			m_timing.bendingGradient = m_timer.elapsedMilliseconds();
			m_timer.start();
			computeHessKappa << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_hessKappas, m_lengths, m_tangents, m_curvatureBinormals, m_materialFrames1, m_materialFrames2, m_kappas);//, d_mats, d_vec, m_restKappas);
			computeBendingGradient << < ivtx_blocksPerGrid, g_threadsPerBlock >> > (
				m_numInnerVtx, m_inner_vtx_to_strand, m_params->m_kb, m_totalJacobian->getMatrix(), m_kappas, m_restKappas, m_invVtxLengths, m_gradKappas, m_hessKappas);//, true, d_localJ);
			m_timing.bendingHessian = m_timer.elapsedMilliseconds();
		}
	}

		//cudaMemcpy(mats.data(), d_mats, mats.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//cudaMemcpy(vec.data(), d_vec, vec.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//cudaMemcpy(grad.data(), d_grad, grad.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
		//cudaMemcpy(localJ.data(), d_localJ, localJ.size() * sizeof(Scalar), cudaMemcpyDeviceToHost);
}

Scalar StrandStates::computeEnergy()
{
	Scalar totalEnergy = 0, energy;

	computeFixingEnergy <<< fixed_blocksPerGrid, g_threadsPerBlock >>> (m_numFixed, m_params->m_ks, m_fixingEnergy, m_fixed_idx, m_vertex_to_strand, m_fixedTargets, m_x);
	CublasCaller<Scalar>::sum(m_cublasHandle, m_numFixed, m_fixingEnergy, &energy);
	totalEnergy += energy;

	computeStretchingEnergy <<< edge_blocksPerGrid, g_threadsPerBlock >>> (m_numEdges, m_params->m_ks, m_stretchingEnergy, m_lengths, m_restLengths);
	CublasCaller<Scalar>::sum(m_cublasHandle, m_numEdges, m_stretchingEnergy, &energy);
	totalEnergy += energy;

	computeTwistingEnergy <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (m_numInnerVtx, m_params->m_kt, m_twistingEnergy, m_twists, m_restTwists, m_invVtxLengths);
	CublasCaller<Scalar>::sum(m_cublasHandle, m_numInnerVtx, m_twistingEnergy, &energy);
	totalEnergy += energy;

	computeBendingEnergy <<< ivtx_blocksPerGrid, g_threadsPerBlock >>> (m_numInnerVtx, m_params->m_kb, m_bendingEnergy, m_kappas, m_restKappas, m_invVtxLengths);
	CublasCaller<Scalar>::sum(m_cublasHandle, m_numInnerVtx, m_bendingEnergy, &energy);
	totalEnergy += energy;

	return totalEnergy;
}