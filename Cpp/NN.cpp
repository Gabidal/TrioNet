#include "../H/NN.h"

NN::NN(int Height, int Width){
    this->Height = Height;
    this->Width = Width;
    for (int i = 0; i < Width * Height; i++){
        Nodes.push_back(new Node());
    }
    //Initialize the input and output nodes, the input node being the first node and the output node being the last node.
    this->Inputs_Node_Indices.push_back(0);
    this->Outputs_Node_Indices .push_back(Height * Width - 1);

    //Generate random connections between the nodes.
    for (int i = 0; i < Height; i++){
        for (int j = 0; j < Width; j++){
            Add_Connection_Random();
        }
    }

    Update_Connections();
}

void NN::Save_Weights(string fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (int i = 0; i < Connections.size(); i++)
    {
        file << Connections[i]->src << "\n" << Connections[i]->dest << "\n" << Connections[i]->weight << "\n";
    }
    file.close();
}

void NN::Load_Weights(string fileName)
{
    std::ifstream file;
    file.open(fileName);
    for (int i = 0; i < Connections.size(); i++)
    {
        file >> Connections[i]->src >> Connections[i]->dest >> Connections[i]->weight;
    }
    file.close();
}

void NN::Add_Connection(int src, int dest, double weight)
{
    Connections.push_back(new Connection(src, dest, weight));
}

void NN::Add_Connection_Random(){
    int src, dest;
    double weight;
    do{
        src = rand() % Height * Width;
        dest = rand() % Height * Width;
    }while(src == dest);
    weight = (double)rand() / RAND_MAX;
    Add_Connection(src, dest, weight);
}

double NN::Sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double NN::Sigmoid_Derivative(double x){
    return x * (1 - x);
}

void NN::Resize_Input_Vector(int Input){
    if (Input > Inputs_Node_Indices.size())
    {
        //Add new inputs to the input node indices vector until it is the same size as the input vector.
        for (int i = Inputs_Node_Indices.size(); i < Input; i++)
        {
            int index = rand() % Height * Width;

            //make sure the input node is not already in the vector.
            bool found = false;
            for (int j = 0; j < Inputs_Node_Indices.size(); j++)
            {
                if (Inputs_Node_Indices[j] == index)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                Inputs_Node_Indices.push_back(index);
            }
        }
    }
}

void NN::Resize_Output_Vector(int Output){
    if (Output > Outputs_Node_Indices.size())
    {
        //Add new outputs to the output node indices vector until it is the same size as the output vector.
        for (int i = Outputs_Node_Indices.size(); i < Output; i++)
        {
            int index = rand() % Height * Width;

            //make sure the output node is not already in the vector.
            bool found = false;
            for (int j = 0; j < Outputs_Node_Indices.size(); j++)
            {
                if (Outputs_Node_Indices[j] == index)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                Outputs_Node_Indices.push_back(index);
            }
        }
    }
}

void NN::Update_Connections(){
    for (auto *i : Connections){
        Nodes[i->dest]->Connections.push_back(i);
    }
}

void Node::Feed_Forward(NN* nn){
    double sum = 0;
    for (auto *i : Connections){
        sum += i->weight * nn->Nodes[i->src]->Value;
    }
    Value = nn->Sigmoid(sum);
}

vector<double> NN::Feed_Forward(vector<double> Inputs)
{
    Resize_Input_Vector(Inputs.size());
    //Set the input nodes to the input values
    for (int i = 0; i < Inputs.size(); i++)
    {
        Nodes[Inputs_Node_Indices[i]]->Value = Inputs[i];
    }
    //Feed forward the information from the connections a node has.
    for (int i = 0; i < Height * Width; i++)
    {
        Nodes[i]->Feed_Forward(this);
    }
    //Return the output nodes
    vector<double> Outputs;
    for (int i = 0; i < Outputs_Node_Indices.size(); i++)
    {
        Outputs.push_back(Nodes[Outputs_Node_Indices[i]]->Value);
    }
    return Outputs;
}

vector<pair<vector<double>, vector<double>>> NN::Get_Training_Data(vector<double> (*Generate_Training_Data)(vector<double>), int in, int out)
{
    Resize_Input_Vector(in);
    Resize_Output_Vector(out);

    vector<pair<vector<double>, vector<double>>> Training_Data;
    for (int i = 0; i < (Height * Width)*Height; i++){
        //Generate a random set of inputs that are passed tot he fnction pointer.
        vector<double> Input;
        vector<double> Output;

        for (int i = 0; i < in; i++)
        {
            Input.push_back((double)rand() / RAND_MAX);
        }
        Output = Generate_Training_Data(Input);
        Training_Data.push_back({Input, Output});
    }
    return Training_Data;
}

void NN::Train(vector<pair<vector<double>, vector<double>>> Training_Data, int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        int Avg = 0;
        for (int j = 0; j < Training_Data.size(); j++)
        {
            Feed_Forward(Training_Data[j].first);
            vector<double> Errors = Back_Propagation(Training_Data[j].second);
            for (auto i : Errors)
                Avg += i;
            Avg /= Errors.size();
        }
        cout << to_string(Avg) << endl;
    }
}

vector<Connection*> NN::Get_Previus_Connections(vector<Connection*> Input)
{
    vector<Connection*> Result;
    for (auto i : Input){
        for (auto c : Connections){
            if (i->src == c->dest){
                Result.push_back(c);
            }
        }
    }
    return Result;
}

vector<double> NN::Back_Propagation(vector<double> Expected_Outputs)
{
    Resize_Output_Vector(Expected_Outputs.size());
    //Calculate the error for each output node
    vector<double> Errors;
    for (int i = 0; i < Outputs_Node_Indices.size(); i++)
    {
        Errors.push_back(Expected_Outputs[i] - Nodes[Outputs_Node_Indices[i]]->Value);
    }
    //find the connections that connect to the output nodes
    vector<Connection*> Previus_Connections;
    for (auto i : Connections){
        for (auto j : Outputs_Node_Indices){
            if (i->dest == j){
                Previus_Connections.push_back(i);
            }
        }
    }
    
    //Go through the Previus_Connections like a recussive function.
    while (true){
        for (auto i : Inputs_Node_Indices){
            for (auto j : Previus_Connections){
                if (j->src == i){
                    break;
                }
            }
        }
        if (Previus_Connections.size() == 0){
            break;
        }

        //Here we calculate the error for each node.
        for (int i = 0; i < Previus_Connections.size(); i++)
        {
            double sum = 0;
            for (int j = 0; j < Previus_Connections.size(); j++)
            {
                sum += Previus_Connections[j]->weight * Errors[j];
            }
            Errors.push_back(sum * Sigmoid_Derivative(Nodes[Previus_Connections[i]->src]->Value));
        }

        Previus_Connections = Get_Previus_Connections(Previus_Connections);
    }

    //Calculate the error for each input node
    for (int i = 0; i < Inputs_Node_Indices.size(); i++)
    {
        double error = 0;
        for (int j = 0; j < Connections.size(); j++)
        {
            if (Connections[j]->src == Inputs_Node_Indices[i])
            {
                error += Connections[j]->error;
            }
        }
        Errors.push_back(error);
    }
    
    //Update the weights
    for (int i = 0; i < Connections.size(); i++)
    {
        Connections[i]->weight += Connections[i]->error * Nodes[Connections[i]->src]->Value;
    }
    return Errors;
}

Connection::Connection(int src, int dest, double weight)
{
    this->src = src;
    this->dest = dest;
    this->weight = weight;
}