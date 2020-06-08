#include "../../H/Task/Task.h"

Task::Task(vector<double>(*function)(vector<double>), int parameter_amount, int min, int max) : Parameter_Amount(parameter_amount)
{
	for (int i = min; i < max;) {
		vector<double> Input;
		for (int j = 0; j < parameter_amount; j++) {
			//make a random value based on the min-max value
			Input.push_back(i);
			Input.push_back(i);
			i++;
		}
		Append(Expected, function(Input));
	}
	//last determine the output size the function gives
	Output_Amount = function({ (double)min + 1 }).size();
}

void Task::Append(vector<double>& dest, vector<double> source)
{
	for (double i : source)
		dest.push_back(i);
}