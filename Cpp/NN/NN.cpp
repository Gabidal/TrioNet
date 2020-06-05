#include "../../H/NN/NN.h"

NN::NN(int Input_Height, int x, int y,  int Output_Height)
{
	//init intput double vector as layer
	Layer* Input = new Layer(false);
	vector<Node> Input_Nodes(Input_Height);
	Input->Nodes = Input_Nodes;
	vector<Node>* Previus = &Input->Nodes;// = Input->Factory(Input_Data);
	Input->Height = Input->Nodes.size();
	Layers.push_back(Input);

	//loop the hidden layer generation
	for (int i = 0; i < y; i++) {
		Layer* Hidden = new Layer(Load_Data_From_File, x);
		Previus = Hidden->Factory(Previus);
		Layers.push_back(Hidden);
	}
	//make the output layer nodes
	Layer* Output = new Layer(Load_Data_From_File, Output_Height);
	Output->Factory(Previus);
	Layers.push_back(Output);
	return;
}

void NN::Load(vector<double> Input) {
	if (Input.size() > Layers.at(0)->Nodes.size()) {
		cout << "Too big input!" << endl;
		exit(1);
	}
	for (int i = 0; i < Input.size(); i++)
		Layers.at(0)->Nodes.at(i).Data = Input.at(i);
	Feed_Foward();
	return;
}

void NN::Train(vector<double> Input_Map, vector<double> Expected_Map)
{
	if ((Input_Map.size() % Layers.at(0)->Height != 0) || (Expected_Map.size() % Layers.at(Layers.size() - 1)->Height != 0)) {
		return;
	}
	while (Input_Map.size() > 0)
	{
		vector<double> Input(Input_Map.begin(), Input_Map.begin() + Layers.at(0)->Height);
		vector<double> Output(Expected_Map.begin(), Expected_Map.begin() + Layers.at(Layers.size() - 1)->Height);
		Load(Input);
		Back_Propagate(Output);
		Input_Map.erase(Input_Map.begin(), Input_Map.begin() + Layers.at(0)->Height);
		Expected_Map.erase(Expected_Map.begin(), Expected_Map.begin() + Layers.at(Layers.size() - 1)->Height);
	}

	for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			for (Connection& z : y.Connections) {
				z.Weight += accumulate(z.Changes.begin(), z.Changes.end(), 0.0) / z.Changes.size();
				z.Changes.clear();
			}
		}
	}
}

void NN::Back_Propagate(vector<double> Expected)
{
	//first clean all errors
	for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			y.Error = 0;
		}
	}
	//first loop every layer backwards
	for (int j = 0; j < Layers.back()->Nodes.size(); j++) {
		Layers.back()->Nodes.at(j).Error = Expected.at(j) - Layers.back()->Nodes.at(j).Data;
	}
	//calculate the error
	for (int i = Layers.size() - 1; i >= 0; i--) {
		for (int j = 0; j < Layers.at(i)->Nodes.size(); j++) {
			Node& Node = Layers.at(i)->Nodes.at(j);
			Node.Delta = Layers.at(i)->Derivate(Node.Data) * Node.Error;
			for (Connection c : Node.Connections) {
				c.Other->Error += Node.Delta * c.Weight;
			}
		}
	}
	//train the weights
	//weight = weight + learning_rate * Delta * input
	for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			for (Connection& z : y.Connections) {
				z.Changes.push_back(0.1 * y.Delta * (z.Other->Data));
			}
		}
	}
}

void NN::Feed_Foward() {
	//the actual feed foward
	for (Layer* i : Layers)
		i->Update();
	return;
}