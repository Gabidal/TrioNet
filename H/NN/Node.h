#ifndef _NODE_H_
#define _NODE_H_
#include <math.h>

class Node
{
public:
	Node(float W) : Weight(W){}
	~Node(){}
	double Activate(double Data);
	double Scale;
	float Weight;
};

#endif