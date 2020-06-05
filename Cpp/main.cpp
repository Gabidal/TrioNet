#include <iostream>
#include "../H/NN/NN.h"
#include <time.h>
using namespace std;

int main() {
	srand(1029384754);
	NN n(2, 5, 4, 1);
	//input		//AND
	for (int i = 0; i < 1; i++)
		n.Train({ 1, 0,
				 1, 1,
				 0, 1,
				 0, 0 },
			{ 0,
			 1,
			 0,
			 0 });
	n.Load({ 1, 0 });
	n.Load({ 1, 1 });
	n.Load({ 0, 1 });
	n.Load({ 0, 0 });
	return 0;
}