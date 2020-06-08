#ifndef _ACTIVATTORS_H_
#define _ACTIVATTORS_H_

#include <numeric>
#include <algorithm>
using namespace std;

namespace Activattors {
	constexpr double e = 2.718281828;//45904523536028747135266249775724709369995;
	double Sigmoid_Activate(double Data);
	double Sigmoid_Derivate(double Data);
	double RELU_Activate(double Data);
	double RELU_Derivate(double Data);
	double Normal_Activate(double Data);
	double Normal_Derivate(double Data);
}

#endif