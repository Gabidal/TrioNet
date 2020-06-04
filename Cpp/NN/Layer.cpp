#include "../../H/NN/Layer.h"
constexpr double e = 2.718281828;//45904523536028747135266249775724709369995;

Layer::Layer(bool new_weights, int height)
{
	Height = height;
	New_Weights = new_weights;
}

Layer::Layer(bool new_weights)
{
	New_Weights = new_weights;
}

Layer::~Layer()
{
}

vector<Node> &Layer::Factory(vector<Node> &Input)
{
	//if we get the nodes as input we need to-
	//create connections with weihts to em and our layer nodes as well
	for (int i = 0; i < Height; i++) {
		Node N; //(insert bias here)
		for (Node& i : Input) {
			//now give the input nodes with new weights-
			//and give em into the node N
			Connection connection(i, Generate_Weight());
			N.Connections.push_back(connection);
		}
		Nodes.push_back(N);
	}
	return Nodes;
}

vector<Node>& Layer::Factory(vector<double> Input)
{	
	//if we get the nodes as input we need to-
	//create connections with weihts to em and our layer nodes as well
	for (int i = 0; i < Input.size(); i++) {
		Node N; //(insert bias here)
		N.Data = Input.at(i);
		Nodes.push_back(N);
	}
	return Nodes;
}

void Layer::Update()
{
	for (Node& i : Nodes)
		if (i.Connections.size() > 0)
			i.Data = Activate(Sum(i));
	return;
}

double Layer::Sum(Node& n)
{
	//first iterate every connection that n has adn multiply them into a new vector where we sum them then.
	double Result = 0.0;
	for (Connection i : n.Connections)
		Result += i.Weight * i.Other.Data;
	return Result;
}

double Layer::Activate(double Sum)
{
	//(1) / (1 + e ^ (-0.01 x)) * 2 - 1
	return (1 / (1 + pow(e, (-Scale * Sum))) * 2 - 1);
}

double Layer::Generate_Weight()
{
	return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
}
