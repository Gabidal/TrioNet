#include "../../H/NN/Layer.h"
constexpr double e = 2.718281828;//45904523536028747135266249775724709369995;

Layer::Layer(bool new_weights, int height)
{
	Height = height;
	New_Weights = new_weights;
}

Layer::~Layer()
{
}

vector<Connection*> Layer::Factory(vector<Connection*> Input)
{
	Generate_Nodes(Input);
	//create the refrence table for nodes
	vector<Connection*> Output;
	for (Node& i : Nodes) {
		Connection* C = new Connection(&i, Generate_Weight());
		Output.push_back(C);
	}
	return Output;
}

void Layer::Update()
{
	for (Node& i : Nodes)
		i.Data = Activate(Sum(&i));
	return;
}

double Layer::Sum(Node* n)
{
	//first iterate every connection that n has adn multiply them into a new vector where we sum them then.
	double Result = 0.0;
	for (Connection i : n->Connections)
		Result += i.Weight * i.Other->Data;
	return Result;
}

double Layer::Activate(double Sum)
{
	//(1) / (1 + e ^ (-0.01 x)) * 2 - 1
	return (1 / (1 + pow(e, (-Scale * Sum))) * 2 - 1);
}

double Layer::Generate_Weight()
{
	return (double)rand() * 500 / 1000;
}

void Layer::Generate_Nodes(vector<Connection*> Input)
{
	//here we will generate the nodes for this chunk
	for (int i = 0; i < Input.size();) {
		Node N;
		for (int j = 0; j < Input.size() / Height; j++) {
			N.Connections.push_back(*Input.at(i));
			if (!(i >= Input.size()))
				i++;
		}
		for (int j = 0; j < Input.size() % Height; j++)
			N.Connections.push_back(*Input.at(i));
		Nodes.push_back(N);
	}

}