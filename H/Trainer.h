#include <vector>
#include <string>
#include <iostream>
#include <thread>

#include "NN.h"

using namespace std;

class Job{
    public:
        vector<double> (*Func)(vector<double>);
        int Input_Count;
        int Output_Count;
        int Complexity;
};

//This class is used to train the neural network in a natural selection way.
//Spawn the given amount of Neural networks and clalculate each of them in seperate thread.
//The best network is the one with the lowerst error score in a iteration count.
//The best network is saved in the file "best.nn"
class Trainer{
    public:
    Job job;
    int Core_Count;
    int Batch_Size;
    int Iteration;
    vector<pair<vector<double>, vector<double>>> Generated_Trainin_Data;
    Trainer(Job job, int Batch_Size, int Iteration);
    void Train();
    void Run_NN(NN* Previus_Best, NN* Current);
};