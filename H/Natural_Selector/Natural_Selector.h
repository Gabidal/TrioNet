#ifndef _NATURAL_SELECTOR_H_
#define _NATURAL_SELECTOR_H_
#include "../NN/NN.h"
#include <thread>
#include <vector>
#include "../Task/Task.h"
using namespace std;

class Natural_Selector
{
public:
	Natural_Selector(Task task, int Amount);
	Natural_Selector(Task task, int Amount, string Starting_FileName);
	void Factory();
	~Natural_Selector();
private:
	void Train(NN*, int);
	Task TASK;
	string Loading_File_Name = "Saved_Weights.txt";
	vector<thread> Trainer;		//there NN's will be trained for about 1000000 cycles
	vector<NN> Nets;
	vector<double> Result_Of_NN;
	void Append(vector<double>& dest, vector<double> source, int i);
};

#endif