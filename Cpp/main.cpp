#include "../H/Trainer.h"

//how many times the data set is looped through
constexpr int ACCURACY = 100;
constexpr int BATCH_SIZE = 1000;

vector<double> Sum(vector<double> in)
{
    return {in[0] + in[1]};
}

int main(int argc, const char** argv){
    /*NN nn(10, 10);
    cout << "Loading Weights" << endl;
    nn.Load_Weights("Saved_Weights.txt");
    cout << "Training AI..." << endl;
    nn.Train(nn.Get_Training_Data(Sum, 2, 1, BATCH_SIZE), ACCURACY);
    cout << "Saving Weights" << endl;
    nn.Save_Weights("Saved_Weights.txt");*/

    Trainer T({ Sum, 2, 1, 10 }, BATCH_SIZE, ACCURACY);

    double a, b, c;

    cin >> a >> b;

    T.Best->Feed_Forward({ a, b }, T.Best->Node_Path);

    cout << T.Best->Nodes[T.Best->Outputs_Node_Indices[0]]->Value << endl;

    int wait;
    cin >> wait;

    return 0;
}