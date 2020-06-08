#include "../../H/Activattors/Activattors.h"

double Activattors::Sigmoid_Activate(double Data)
{
	//(1) / (1 + e ^ (-0.01 x)) * 2 - 1
	//return (1 / (1 + pow(e, (-Sensitivity * Sum))));
	return 1 / (1 + pow(e, -Data));
	//return Sum < 0 ? 0 : Sum;
}

double Activattors::Sigmoid_Derivate(double Data)
{
	//(2e^(-Sensitivity*Data)Sensitivity)/((e^(-Sensitivity*Data)+1)^2)
	//return (2*pow(e, -Sensitivity*Data)*Sensitivity)/(pow(pow(e, -Sensitivity*Data)+1, 2));
	return Data * (1.0 - Data);
}

double Activattors::RELU_Activate(double Data)
{
	return Data < 0 ? 0 : Data;
}

double Activattors::RELU_Derivate(double Data)
{
	if (Data < 0)
		return 0;
	return 1;
}

double Activattors::Normal_Activate(double Data)
{
	return Data;
}

double Activattors::Normal_Derivate(double Data)
{
	return 1;
}
