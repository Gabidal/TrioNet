#ifndef _CONNECTION_H_
#define _CONNECTION_H_
#include <vector>
#include "Node.h"
using namespace std;

class Connection
{
public:
	Connection(Node l, Node r) : Left(l), Right(r){}
	~Connection(){}
	Node Left;
	Node Right;
};

#endif