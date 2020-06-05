#ifndef _CONNECTION_H_
#define _CONNECTION_H_
#include <vector>
class Node;

using namespace std;

class Connection
{
public:
	Connection(Node* O, double w) : Other(O), Weight(w) {}
	~Connection(){}
	double Weight;
	Node* Other;
	vector<double> Changes;
};

#endif