#ifndef _NODE_H_
#define _NODE_H_
#include <math.h>
#include <vector>
#include "Connection.h"

using namespace std;

class Node
{
public:
	Node() {}
	Node(double bias) : Bias(bias) {}
	~Node() {}
	double Bias = 0;
	double Data = 1;
	double Delta = 0;
	double Error = 0;
	vector<Connection> Connections;
};

#endif