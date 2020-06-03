#ifndef _NODE_H_
#define _NODE_H_
#include <math.h>
#include <vector>
#include "Connection.h"

using namespace std;

class Node
{
public:
	Node(){}
	~Node(){}
	double Bias;
	double Data;
	vector<Connection> Connections;
};

#endif