
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

			m_batchSize = batch_size;
			m_group_ = 1;

			m_bottom_dim_ = 0;// default batch =1
			m_top_dim_ = 0;

			m_winogradWeight = NULL;
			m_winogradInput = NULL;


			// Output width.
			m_oW = (m_iW + m_pad * 2 - m_kW) / m_sW + 1;
			m_oH = (m_iH + m_pad * 2 - m_kH) / m_sH + 1;

			if (alg == WT_8X8_F_6X6_3X3) {

				tile_h_in_ = 8;
				tile_w_in_ = 8; /* input tile size */

				tile_h_out_ = tile_h_in_ - m_kH + 1;
				tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */

				ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
				ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;

			}
			else if (alg == WT_6X6_F_4X4_3X3) {

				tile_h_in_ = 6;
				tile_w_in_ = 6; /* input tile size */

				tile_h_out_ = tile_h_in_ - m_kH + 1;
				tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */

				ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
				ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;

			}
			else throw("convolution algorithm error!");

		}

		template <typename Dtype>
		const std::shared_ptr<Dtype> get_inference_cpu(Dtype* data, const Dtype* par, Dtype* col_buff) {

			m_inputOrg = data;
			m_weightOrg = par;
			m_col_buff = col_buff;


			std::shared_ptr<Dtype> resOut = std::shared_ptr<Dtype>(new Dtype[m_oH*m_oW*conv_out_channels_]);

			//trans weight to winograd domain
			trans_weight2wiongrad();


			for (int n = 0; n < m_batchSize; n++) {

				//trans input to winograd domain
				trans_input2winograd(m_inputOrg + n*m_bottom_dim_, m_col_buff);


				// Convolution in Winograd domain
				winograd_conv();


				// Transform back to time domain	
				trans2spatial(resOut.get() + n*this->m_top_dim_);

				//bias
				if (this->m_bias) {

					int base = conv_in_channels_ * conv_out_channels_ * m_kW * m_kH;

					const Dtype* bias = &par[base];

					this->forward_cpu_bias(resOut.get() + n * this->m_top_dim_, bias);
				}
			}

			return  resOut;
		}


	public:
		~WinogradLayer() {