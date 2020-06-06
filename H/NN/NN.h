#ifndef _NN_H_
#define _NN_H_
#include "Layer.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>

using namespace std;

class NN
{
public:
	//input height, Hidden layer Y, Hidden layer X, otput size
	//also generate the layers
	NN(int, int, int, int);
	~NN(){}
	void Generate_Data_Set(vector<double>(*function) (vector<double>), int Parameter_amount, int min, int max);
	vector<double> Train();
	void Test(vector<double>);
	double Output(int i);
	void Save();
	void Load();
	void Clean_Data_Set();
private:
	void Append(vector<double>&, vector<double>);
	int Input_Format = 0;
	int Output_Format = 0;
	int Get_Exponent_Value(double);
	double Back_Propagate(vector<double>);
	void Feed_Foward();
	//0:th has the input && last has the output, layer
	bool Load_Data_From_File = false;
	vector<Layer*> Layers;
	vector<double> Input_Map;
	vector<double> Expected_Map;
};

#endif