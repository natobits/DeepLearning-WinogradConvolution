
#ifndef WINOGRAD_LAYER_H
#define WINOGRAD_LAYER_H

#include <memory>
#include "winograd_kernel.h"
#include "tool.h"
//winograd for cpu inference

// default 3x3
const int KERNEL_SIZE = 3;

namespace WINOGRAD_KERNEL
{

	template <typename Dtype>
	class WinogradLayer {

	private:

		int m_group_;
		int m_batchSize;

		int m_bottom_dim_;// par size
		int m_top_dim_;

		// The following variables are initialized in WeightAlign
		int tile_h_in_, tile_w_in_; /* input tile size */
		int tile_h_out_, tile_w_out_; /* output tile size */
		int ntiles_h_, ntiles_w_; /* number of tiles */
