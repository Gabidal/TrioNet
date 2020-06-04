#include <iostream>
#include "../H/NN/Layer.h"
using namespace std;

int main() {
	//input 
	Layer Start(false);
	vector<Node> Input = Start.Factory({ 1, 0, 1, 1, 0, 1, 1, 1, 0, 0});
	Layer Hidden1(false, 5);
	vector<Node> Hidden = Hidden1.Factory(Input);
	Layer End(false, 2);
	vector<Node> Output = End.Factory(Hidden);

	vector<Layer> NN;

	NN.push_back(Start);
	NN.push_back(Hidden1);
	NN.push_back(End);

	//the actual feed foward
	for (Layer& i : NN) {
		i.Update();
	}
	return 0;
}