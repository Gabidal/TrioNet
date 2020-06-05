#ifndef _NN_H_
#define _NN_H_
#include "Layer.h"
#include <vector>
#include <iostream>

using namespace std;

class NN
{
public:
	//input height, Hidden layer Y, Hidden layer X, otput size
	//also generate the layers
	NN(int, int, int, int);
	~NN(){}
	void Load(vector<double>);
	void Train(vector<double>);
private:
	void Feed_Foward();
	//0:th has the input && last has the output, layer
	bool Load_Data_From_File = false;
	vector<Layer*> Layers;
};

#endif