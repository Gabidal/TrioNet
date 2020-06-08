#ifndef _TASK_H_
#define _TASK_H_

#include <vector>

using namespace std;

class Task
{
public:
	Task(vector<double>(*function) (vector<double>), int parameter_amount, int min, int max);
	~Task();
	int Parameter_Amount = 0;
	int Output_Amount = 0;
	vector<double> Input;
	vector<double> Expected;
	void Append(vector<double>& dest, vector<double> source);
private:
};

#endif