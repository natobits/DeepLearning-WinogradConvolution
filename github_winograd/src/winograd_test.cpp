
#pragma once

#include "../include/winograd_kernel.h"
#include "../include/winograd_layer.h"
#include <iostream>
#include <iomanip>

using namespace WINOGRAD_KERNEL;
using namespace std;

const int CIN = 3;
const int COUT = 7;

const int IH = 25;
const int IW = 25;

const int PRECISE = 0;

#define INPUT_INTEGER 1
#define KERNEL_INTEGER 1

void testWinograd();

int main() {


	WINOGRAD_KERNEL::winograd2D_initialize();

	testWinograd();

	return 0;
}

void testWinograd() {

	//int batch_size = 1;

	int tiH = IH;
	int tiW = IW;

	int tkW = 3;
	int tkH = 3;

	int tsW = 1;
	int tsH = 1;

	int tiC = CIN;
	const int toC = COUT;

	bool tbias = true;

	int tpad = 1;

	const auto toH = (tiH + tpad * 2 - tkH) / tsH + 1;

	// Output width.
	const auto toW = (tiW + tpad * 2 - tkW) / tsW + 1;
