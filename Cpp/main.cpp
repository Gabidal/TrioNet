#include "../H/NN.h"

//how many times the data set is looped through
constexpr int ACCURACY = 10;
constexpr int BATCH_SIZE = 100;

vector<double> Sum(vector<double> in)
{
    return {in[0] + in[1]};
}

int main(int argc, const char** argv){
    NN nn(2, 2);
    nn.Load_Weights("Saved_Weights.txt", argv);
    nn.Train(nn.Get_Training_Data(Sum, 2, 1, BATCH_SIZE), ACCURACY);
    nn.Save_Weights("Saved_Weights.txt", argv);
}