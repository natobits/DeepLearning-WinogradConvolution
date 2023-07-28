
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

		int conv_in_channels_; //ic
		int conv_out_channels_;//oc

		int m_iH;
		int m_iW;

		int m_oH;
		int m_oW;

		int m_kH;
		int m_kW;
		int m_sH;
		int m_sW;

		int m_pad;
		bool m_bias;

	private:

		Dtype* m_inputOrg;
		const Dtype* m_weightOrg;

		Dtype* m_winogradWeight; // support NCHW storage
		Dtype* m_winogradInput;

		Dtype* m_col_buff;//buffer

		WINOGRAD_ALG m_alg;

	public:

		WinogradLayer(WINOGRAD_ALG alg, int batch_size, int iH, int iW, int iC, int kH, int kW, int sH, int sW, int oC, int pad, bool bias = true) : m_alg(alg) {

#if DEBUG_WINOGRAD
			assert(kH == kW, "kernel 3x3 is the best choice, some errors may occur for other kernels");
#endif
			m_iH = iH;
			m_iW = iW;
			conv_in_channels_ = iC;
			m_kH = kH;
			m_kW = kW;
			m_sH = sH;
			m_sW = sW;
			conv_out_channels_ = oC;
			m_pad = pad; // pad_h = pad_w
			m_bias = bias;
