
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