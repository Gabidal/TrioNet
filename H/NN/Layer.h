#ifndef _LAYER_H_
#define _LAYER_H_
#include "Node.h"
#include "Connection.h"

#include <vector>

using namespace std;

class Layer
{
public:
	Layer(bool new_weights, int height);
	~Layer();
	//this factory creates a pipeline to work with
	vector<Connection*> Factory(vector<Connection*>);
	//for raw input
	vector<Connection*> Factory(vector<double>);
	//this function can start line compute the NN matrix from this chunk
	void Update();

private:
	int Height;
	bool New_Weights;
	double Scale = 1;
	vector<Node> Nodes;
	//map<double, vector<Connection*>> Previus_Usage;
	double Sum(Node*);
	double Activate(double Data);
	double Generate_Weight();
	void Generate_Nodes(vector<Connection*> Input);
	//for raw input
	void Generate_Nodes(vector<double> Input);
};

#endif // !_CHUNK_H_