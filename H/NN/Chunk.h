#ifndef _CHUNK_H_
#define _CHUNK_H_
#include "Node.h"
#include <vector>

class Chunk
{
public:
	Chunk();
	~Chunk();
	double Sum();
	vectro<Node> Factory(vector<Node>);
	vector<Node> Nodes;

private:

};

#endif // !_CHUNK_H_