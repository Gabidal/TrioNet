#include "../../H/NN/Node.h"

inline void Node::Give_Data(float D)
{
	Data = D;
}

inline float Node::Get_Data()
{
	return Data;
}

inline void Node::Update_Weight(float W)
{
	Weight = W;
}

inline float Node::Get_Weight()
{
	return Weight;
}
