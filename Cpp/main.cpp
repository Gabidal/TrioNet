#include "../H/NN.h"

//how many times the data set is looped through
constexpr int ACCURACY = 10;

vector<double> Sum(vector<double> in)
{
    return {in[0] + in[1]};
}

int main(int argc, const char** argv){
    NN nn(10, 10);
    nn.Load_Weights("Saved_Weights.txt", argv);
    nn.Train(nn.Get_Training_Data(Sum, 2, 1), ACCURACY);
    nn.Save_Weights("Saved_Weights.txt", argv);
}