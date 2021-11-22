#include "../H/NN.h"

constexpr double ACCURACY = 0.01;

vector<double> Sum(vector<double> in)
{
    return {in[0] + in[1]};
}

int main(){
    NN nn(10, 10);
    nn.Train(nn.Get_Training_Data(Sum, 2, 1), ACCURACY);
    nn.Save_Weights("Saved_Weights.txt");
}