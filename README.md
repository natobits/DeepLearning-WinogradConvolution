# DeepLearning-WinogradConvolution
DeepLearning-WinogradConvolution is a Winograd based kernel for convolutions in deep learning frameworks. It's an implementation of Winograd convolutions that supports three WT methods, namely, WT_6X6_F_4X4_3X3, WT_8X8_F_4X4_5X5, and WT_8X8_F_6X6_3X3, where a 3x3 convolution kernel is recommended. This project borrows some components from SkimCaffe, but the Winograd kernel implemented here is more portable.

## Dependencies
For performance reasons, a fast blas such as mkl-gemm or openblas is preferred.

## Building
The project only requires header files written in C++ and supports both Windows and Linux. This version was built on VS 2015.

## Testing
You can refer to winograd_test.cpp for testing.

## Packaging
The header file 'include/winograd_layer.h' can be integrated natively into some popular deep learning frameworks a