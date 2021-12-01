#include "../H/Trainer.h"

Trainer::Trainer(Job job, int Batch_Size, int Iteration){
    //First generate a batch of Neural Networks.
    this->job = job;
    this->Core_Count = thread::hardware_concurrency() / 2;
    this->Batch_Size = Batch_Size;
    this->Iteration = Iteration;
    Train();
}

void Trainer::Run_NN(NN* Previus_Best, NN* Current)
{
    if (Previus_Best) {
        Current->Inputs_Node_Indices = Previus_Best->Inputs_Node_Indices;
        Current->Outputs_Node_Indices = Previus_Best->Outputs_Node_Indices;

        //copy the connection information from the previus NN to this one
        for (auto& i : Previus_Best->Connections) {
            Current->Connections.push_back(new Connection(i->src, i->dest, i->weight));
        }

    }

    Current->Start_Train(Generated_Trainin_Data, Iteration);
}

void Trainer::Train()
{
    NN DULL(1, 1);

    //Generate the training data.
    Generated_Trainin_Data = DULL.Get_Training_Data(job.Func, job.Input_Count, job.Output_Count, Batch_Size);

    //start the training, after a certain iteration look for best candidate to save and load the next batch based on this one.
    NN* Best = new NN(job.Complexity, job.Complexity);
    Best->Resize_Input_Vector(job.Input_Count);
    Best->Resize_Output_Vector(job.Output_Count);

    //Load the previus best trained weights.
    Best->Load_Weights("Saved_Weights.txt");

    vector<thread> Threads;
    //Allocate the cores
    for (int i = 0; i < Core_Count; i++) {
        Threads.push_back(thread());
    }

    while (Best->Lowest_Error > 0) {
        vector<NN*> Candidates;
        //Train the NN
        for (auto& T : Threads) {
            NN* Candidate = new NN(job.Complexity, job.Complexity);
            Candidates.push_back(Candidate);

            T = thread(&Trainer::Run_NN, this, Best, Candidate);
            //thread([this, Best, Candidate] { Run_NN(Best, Candidate); });
        }

        cout << "Error: " << Best->Lowest_Error << endl;
    }

    for (auto& T : Threads) {
        while (T.joinable() == false) {
            _sleep(500);
        }
        T.join();
    }

    Best->Save_Weights("Saved_Weights.txt");

}