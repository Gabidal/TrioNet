#ifndef _NODE_H_
#define _NODE_H_

class Node
{
public:
	Node(float W) : Weight(W){}
	~Node(){}
	//data handling
	void Give_Data(float);
	float Get_Data();
	//Relativeimportnce handling
	void Update_Weight(float);
	float Get_Weight();
private:
	float Data;
	float Weight;
};

#endif