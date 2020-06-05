#ifndef _CONNECTION_H_
#define _CONNECTION_H_
class Node;

class Connection
{
public:
	Connection(Node* O, double w) : Other(O), Weight(w) {}
	~Connection(){}
	double Weight;
	Node* Other;
};

#endif