#include "../H/NN.h"

//how many times the data set is looped through
constexpr int ACCURACY = 1;
constexpr int BATCH_SIZE = 10000;

vector<double> Sum(vector<double> in)
{
    return {in[0] + in[1]};
}

int main(int argc, const char** argv){
    NN nn(10, 10);
    cout << "Loading Weights" << endl;
    nn.Load_Weights("Saved_Weights.txt", argv);
    cout << "Training AI..." << endl;
    nn.Train(nn.Get_Training_Data(Sum, 2, 1, BATCH_SIZE), ACCURACY);
    cout << "Saving Weights" << endl;
    nn.Save_Weights("Saved_Weights.txt", argv);

    return 0;
}