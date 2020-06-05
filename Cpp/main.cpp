#include <iostream>
#include "../H/NN/NN.h"
using namespace std;

int main() {
	NN n(2, 3, 2, 1);
	//input 
	for (int i = 0; i < 5000; i++) {
		n.Load({ 1, 1 });	//AND
		n.Train({ 1 });
	}
	n.Load({ 1, 0 });	//XOR
	return 0;
}