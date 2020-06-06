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

void NN::Test(vector<double> Input) {
	if (Input.size() > Layers.at(0)->Nodes.size()) {
		cout << "Too big input!" << endl;
		exit(1);
	}
	for (int i = 0; i < Input.size(); i++)
		Layers.at(0)->Nodes.at(i).Data = Input.at(i) / Input_Format;
	Feed_Foward();
	return;
}

void NN::Append(vector<double>& dest, vector<double> source)
{
	for (double i : source)
		dest.push_back(i);
}

double NN::Output(int i)
{
	return Layers.back()->Nodes.at(i).Data * Output_Format;
}

void NN::Save()
{
	system("del Saved_Weights.txt");
	ofstream File("Saved_Weights.txt");
	for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			for (Connection& z : y.Connections) {
				File << z.Weight << endl;
			}
		}
	}
	File.close();
}

void NN::Load()
{
	ifstream File("Saved_Weights.txt");
	if (File.is_open()) {
		for (Layer* x : Layers) {
			for (Node& y : x->Nodes) {
				for (Connection& z : y.Connections) {
					string Line = "";
					getline(File, Line);
					z.Weight = atof(Line.c_str());
				}
			}
		}
	}
	else {
		cout << "No Saved Weights found. Generating new ones." << endl;
	}
	return;
}

void NN::Clean_Data_Set()
{
	Input_Map.clear();
	Expected_Map.clear();
}

void NN::Generate_Data_Set(vector<double>(*function) (vector<double>), int Parameter_amount, int min, int max)
{
	for (int i = min; i < max;) {
		vector<double> Input;
		for (int j = 0; j < Parameter_amount; j++) {
			//make a random value based on the min-max value
			Input_Map.push_back(i);
			Input.push_back(i);
			i++;
		}
		Append(Expected_Map, function(Input));
	}
	double biggestInput = *max_element(Input_Map.begin(), Input_Map.end());
	double biggestOuput = *max_element(Expected_Map.begin(), Expected_Map.end());

	double Divider = pow(10, Get_Exponent_Value(biggestInput));
	Input_Format = Divider;
	//for (int i = 0; i < Input_Map.size(); i++)
	//	Input_Map.at(i) /= Divider;

	Divider = pow(10, Get_Exponent_Value(biggestOuput));
	Output_Format = Divider;
	for (int i = 0; i < Expected_Map.size(); i++)
		Expected_Map.at(i) /= Divider;
}

double NN::Train()
{
	if ((Input_Map.size() % Layers.at(0)->Height != 0) || (Expected_Map.size() % Layers.at(Layers.size() - 1)->Height != 0) ||
		(Input_Map.size() / Layers.at(0)->Height) != (Expected_Map.size() / Layers.at(Layers.size() - 1)->Height)) {
		return -1;
	}
	double Average_Percentage_Error = 0;
	int Input_Iterator = 0;
	int Output_Iterator = 0;
	while (Input_Iterator < Input_Map.size())
	{
		vector<double> Input(Input_Map.begin() + Input_Iterator, Input_Map.begin() + Layers.at(0)->Height + Input_Iterator);
		vector<double> Output(Expected_Map.begin() + Output_Iterator, Expected_Map.begin() + Layers.at(Layers.size() - 1)->Height + Output_Iterator);
		Test(Input);
		Average_Percentage_Error += Back_Propagate(Output);
		Input_Iterator += Layers.at(0)->Height;
		Output_Iterator += Layers.at(Layers.size() - 1)->Height;
	}

	/*for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			for (Connection& z : y.Connections) {
				z.Weight += accumulate(z.Changes.begin(), z.Changes.end(), 0.0) / z.Changes.size();
				z.Changes.clear();
			}
		}
	}*/
	return Average_Percentage_Error / (Input_Map.size() / Layers.at(0)->Height);
}

int NN::Get_Exponent_Value(double value)
{
	int i = 0;
	while (value > 1) {
		value /= 10;
		i++;
	}
	return i;
}

double NN::Back_Propagate(vector<double> Expected)
{
	double Average_Percentage_Error = 0;
	//first clean all errors
	for (Layer* x : Layers) {
		for (Node& y : x->Nodes) {
			y.Error = 0;
		}
	}
	//first loop every last layer backwards
	for (int j = 0; j < Layers.back()->Nodes.size(); j++) {
		Layers.back()->Nodes.at(j).Error = Expected.at(j) - Layers.back()->Nodes.at(j).Data;
		Average_Percentage_Error += abs(Layers.back()->Nodes.at(j).Error / Expected.at(j));
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
				//z.Changes.push_back(0.1 * y.Delta * (z.Other->Data));
				z.Weight += 0.1 * y.Delta * z.Other->Data;
			}
		}
	}
	return Average_Percentage_Error / Layers.back()->Nodes.size();
}

void NN::Feed_Foward() {
	//the actual feed foward
	for (Layer* i : Layers)
		i->Update();
	return;
}