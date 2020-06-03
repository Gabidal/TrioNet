#include "../../H/NN/Node.h"
constexpr double e = 2.718281828//45904523536028747135266249775724709369995;

double Node::Activate(double Data)
{
	//(1) / (1 + e ^ (-0.01 x)) * 2 - 1
	return (1/(1+pow(e, (-Scale * Data))) * 2 - 1);
}
