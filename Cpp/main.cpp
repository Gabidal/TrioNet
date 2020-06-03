#include <iostream>
#include "../H/NN/Layer.h"
using namespace std;

int main() {
	//input 
	Layer Start(false, 10);
	vector<Connection*> Input = Start.Factory({ 1, 0, 1, 1, 0, 1, 1, 1, 0, 0});
	vector<Connection*> Output;
	vector<Layer> NN;
	//first make a NN pipeline
	for (int i = 0; i < 10; i++) {
		//on second loop time it breaks
		Layer Hidden(false, 5);
		Output = Hidden.Factory(Input);
		NN.push_back(Hidden);
		Input = Output;
	}

	//the actual feed foward
	for (Layer& i : NN) {
		i.Update();
	}
	return 0;
}