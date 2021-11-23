#include "../H/NN.h"

constexpr double Learning_Rate = 0.01;

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

void NN::Save_Weights(string fileName, const char** argv)
{
    std::fstream file;

    string Dir = string(argv[0]);
    int Last_Slash = Dir.find_last_of('\\');

    Dir = Dir.substr(0, Last_Slash + 1);

    file.open( Dir + fileName);
    if (!file.is_open()) {
        cout << "Cant open file " + fileName << endl;
    }
    else {
        file << Connections.size();
        for (int i = 0; i < Connections.size(); i++)
        {
            file << Connections[i]->src << "\n" << Connections[i]->dest << "\n" << Connections[i]->weight << "\n";
        }
    }
    file.close();
}

void NN::Load_Weights(string fileName, const char** argv)
{
    std::fstream file;

    string Dir = string(argv[0]);
    int Last_Slash = Dir.find_last_of('\\');

    Dir = Dir.substr(0, Last_Slash + 1);

    file.open(Dir + fileName);
    if (!file.is_open()) {
        cout << "Cant open file " + fileName << endl;
    }
    else {
        int Connections_Size = 0;
        file >> Connections_Size;
        Connections = vector<Connection*>(Connections_Size);
        for (auto &i : Connections) {
            i = new Connection();
        }
        for (int i = 0; i < Connections.size(); i++)
        {
            file >> Connections[i]->src >> Connections[i]->dest >> Connections[i]->weight;
        }
    }
    file.close();
}

void NN::Add_Connection(int src, int dest, double weight)
{
    Connections.push_back(new Connection(src, dest, weight));
    Nodes[dest]->Connections.push_back(Connections.back());
}

void NN::Add_Connection_Random(){
    int src, dest;
    double weight;
    do{
        src = rand() % (Height * Width);
        dest = rand() % (Height * Width);
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
    //first clean all the node connections.
    for (auto* i : Nodes) {
        i->Connections.clear();
    }
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
        double Previus_Avg = 0;
        double Avg = 0;
        for (int j = 0; j < Training_Data.size(); j++)
        {
            Feed_Forward(Training_Data[j].first);
            vector<double> Errors = Back_Propagation(Training_Data[j].second);
            for (auto i : Errors)
                Avg += i;
            Avg /= Errors.size();
            if (Previus_Avg == Avg) {
                //There are no connections between the input and output nodes.
                Add_Connection_Random();
            }
            Previus_Avg = Avg;
        }
        cout << "Error: " + to_string(Avg) + "%" << endl;
    }
}

vector<Node*> Trace;
vector<Node*> NN::Find_Path(Connection* c)
{
    vector<Node*> Result;

    for (auto i : Trace) {
        if (i == Nodes[c->dest]) {
            return Result;
        }
    }

    //Check if we have hit the end
    for (auto i : Inputs_Node_Indices) {
        if (c->src == i) {
            Result.push_back(Nodes[i]);
            return Result;
        }
    }

    vector<Connection*> Connections = Nodes[c->src]->Connections;

    Trace.push_back(Nodes[c->dest]);
    for (auto *i : Connections) {
        for (auto j : Find_Path(i)) {
            Result.push_back(j);
        }
    }
    Trace.pop_back();

    if (Result.size()) {
        Result.push_back(Nodes[c->dest]);
    }

    return Result;
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
    //Make sure the size of the expected outputs vector is the same as the number of outputs.
    //We dont want to shrink the output nor input node count, this is because the left over nodes are experiences.
    Resize_Output_Vector(Expected_Outputs.size());

    //Try to path find through the connections, a conneciton that starts from a certain input node, and end in a certain output node.
    //Gather these connections into a list, this list will be back propagated.
    vector<vector<Node*>> Node_Path;

    for (auto Output : Outputs_Node_Indices) {
        vector<Node*> Single_Path;
        bool No_Connection = true;
        while (No_Connection){
            for (auto Con : Nodes[Output]->Connections) {
                for (auto Path : Find_Path(Con)) {
                    Single_Path.push_back(Path);
                }
            }
            if (Single_Path.size() == 0){
                //This means that there were no connections from the output node to the input node.
                //Thus we need to create a random connection.
                Add_Connection_Random();
            }
            else{
                No_Connection = false;
            }
        }
        Node_Path.push_back(Single_Path);
    }

    //Now that we have the path, we can back propagate the errors.
    //After that we can update the weights based on the errors.
    vector<double> Errors;
    for (int i = 0; i < Node_Path.size(); i++) {
        for (int j = 0; j < Node_Path[i].size(); j++) {
            //The error is the difference between the expected output and the actual output.
            double Error = Expected_Outputs[i] - Node_Path[i][j]->Value;
            Errors.push_back(Error);
        }
        //The error is the sum of the errors in the path.
        double Error = 0;
        for (auto i : Errors) {
            Error += i;
        }
        //The error is the error of the node.
        Errors.push_back(Error);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 1]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 2]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 3]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 4]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 5]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 6]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 7]->Value);
        //The error is the error of the node.
        Errors.push_back(Node_Path[i][Node_Path[i].size() - 8]->Value);
        //The error is the error of the node.
    }
    return Errors;
}

Connection::Connection(int src, int dest, double weight)
{
    this->src = src;
    this->dest = dest;
    this->weight = weight;
}