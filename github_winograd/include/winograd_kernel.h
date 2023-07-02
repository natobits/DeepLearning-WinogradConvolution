

#ifndef WINOGRAD_KERNEL_H
#define WINOGRAD_KERNEL_H

#include <memory>

#define DEBUG_WINOGRAD 1

#if DEBUG_WINOGRAD
	#include <cassert>
#endif

namespace WINOGRAD_KERNEL {

	const enum WINOGRAD_MATRIX {
		WINOGRAD_A = 0,
		WINOGRAD_B,
		WINOGRAD_G,
	};
	const enum WINOGRAD_ALG {
		WT_8X8_F_6X6_3X3 = 0,
		WT_6X6_F_4X4_3X3,
		WT_8X8_F_4X4_5X5,
	};

	const int MATRIX_KINDS = 3;
	const int WINOGRAD_PAIR_KINDS = 3;

	template<WINOGRAD_ALG a>
	struct WinogradTransformMatrix {};

	/**
	* compute Kronecker product of in1 and in2, where in1 is a m by n matrix and in2 is a p by q matrix
	*
	* @params out an (m*p) by (n*q) matrix stored in row major
	* @params in1 an m by n matrix stored in row major
	* @params in2 an p by q matrix stored in row major
	*/
	void kronecker_product(float *out, const float *in1, const float *in2, int m, int n, int p, int q);

	//singleton, precomputation before inference  
	void winograd2D_initialize();

	template<>
	struct WinogradTransformMatrix<WT_6X6_F_4X4_3X3>
	{
		// wt6x6, F(4x4,3x3)
	private: