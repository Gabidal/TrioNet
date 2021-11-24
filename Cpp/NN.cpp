#include "../H/NN.h"

constexpr double Learning_Rate = 0.0001;

NN::NN(int Height, int Width){
    this->Height = Height;
    this->Width = Width;
    for (int i = 0; i < Width * Height; i++){
        Nodes.push_back(new Node());
    }
}

void NN::Save_Weights(string fileName)
{
    std::ofstream file("../../" + fileName, std::ios::binary);
    if (!file.is_open()) {
        cout << "Cant open file " + fileName << endl;
    }
    else {
        string Buffer = to_string(Connections.size()) + "\n";
        for (int i = 0; i < Connections.size(); i++)
        {
            Buffer += to_string(Connections[i]->src) + "\n" + to_string(Connections[i]->dest) + "\n" + to_string(Connections[i]->weight) + "\n";
        }
        file.write(Buffer.data(), Buffer.size());
    }
    file.close();
}

void NN::Load_Weights(string fileName)
{
    std::ifstream file("../../" + fileName, std::ios::binary);
    if (!file.is_open()) {
        cout << "Cant open file " + fileName << endl;
    }
    else {
        file.seekg(0, ios_base::end);
        int Connections_Size = 0;
        int File_Size = file.tellg();
        file.seekg(std::ios::beg);
        string Buffer(File_Size + 1, '\0');

        if (File_Size > 0) {
            file.read((char*)Buffer.data(), File_Size);

            vector<string> Elements;
            int Last_New_Line_Index = 0;
            for (int i = 0; i < Buffer.size(); i++) {
                if (Buffer[i] == '\n') {
                    Elements.push_back(Buffer.substr(Last_New_Line_Index, i - Last_New_Line_Index));
                    Last_New_Line_Index = i;
                }
            }

            //the first element is the size of the weights.
            Connections = vector<Connection*>(atoi(Elements[0].c_str()));
            for (auto& i : Connections) {
                i = new Connection();
            }

            for (int i = 1; i < Elements.size(); i += 3) {
                Connections[i / 3]->src = atoi(Elements[i].c_str());
                Connections[i / 3]->dest = atoi(Elements[i + 1].c_str());
                Connections[i / 3]->weight = atof(Elements[i + 2].c_str());

                Nodes[Connections[i / 3]->dest]->Connections.push_back(Connections[i / 3]);
            }
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
    weight = ((double)rand() / (RAND_MAX)) * 2 - 1;
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
    double sum = 1;
    for (auto *i : Connections){
        sum += i->weight * nn->Nodes[i->src]->Value - 1;
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

vector<pair<vector<double>, vector<double>>> NN::Get_Training_Data(vector<double> (*Generate_Training_Data)(vector<double>), int in, int out, int Batch)
{
    Resize_Input_Vector(in);
    Resize_Output_Vector(out);

    vector<pair<vector<double>, vector<double>>> Training_Data;
    for (int i = 0; i < Batch; i++){
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

void NN::Train(vector<pair<vector<double>, vector<double>>> Training_Data, int Iterations)
{
    vector<vector<Connection*>> Node_Path;
    //Try to path find through the connections, a conneciton that starts from a certain input node, and end in a certain output node.
    //Gather these connections into a list, this list will be back propagated.
    for (auto Output : Outputs_Node_Indices) {
        vector<Connection*> Single_Path;
        bool No_Connection = true;
        while (No_Connection) {
            for (auto Con : Nodes[Output]->Connections) {
                for (auto Path : Find_Path(Con)) {
                    Single_Path.push_back(Path);
                }
            }
            if (Single_Path.size() == 0) {
                //This means that there were no connections from the output node to the input node.
                //Thus we need to create a random connection.
                Add_Connection_Random();
            }
            else {
                No_Connection = false;
            }
        }
        Node_Path.push_back(Single_Path);
    }
    for (int i = 0; i < Iterations; i++)
    {
        double Avg = 0;
        for (int j = 0; j < Training_Data.size(); j++)
        {
            Feed_Forward(Training_Data[j].first);
            vector<double> Errors = Back_Propagation(Training_Data[j].second, Node_Path);
            for (auto i : Errors)
                Avg += i;
            Avg /= Errors.size();
        }
        cout << "Error: " + to_string(Avg) + "%" << endl;
    }
    cout << "All connections count: " << Connections.size() << endl;
    Clean_Dum_Connections();
    cout << "True connections count: " << Connections.size() << endl;
}

vector<Connection*> Trace;
vector<Connection*> NN::Find_Path(Connection* c)
{
    vector<Connection*> Result;

    for (auto i : Trace) {
        if (i == c) {
            return Result;
        }
    }

    //Check if we have hit the end
    for (auto i : Inputs_Node_Indices) {
        if (c->src == i) {
            Result.push_back(c);
            return Result;
        }
    }

    vector<Connection*> Connections = Nodes[c->src]->Connections;

    Trace.push_back(c);
    for (auto *i : Connections) {
        if (i->Dum)
            continue;
        for (auto j : Inputs_Node_Indices) {
            int Self_Distance = abs(j - c->src);
            int Connection_Distance = abs(j - i->src);

            if (Connection_Distance < Self_Distance) {
                for (auto k : Find_Path(i)) {
                    Result.push_back(k);
                }
            }
            else {
                i->Dum = true;
            }
        }
    }
    Trace.pop_back();

    if (Result.size()) {
        Result.push_back(c);
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

vector<double> NN::Back_Propagation(vector<double> Expected_Outputs, vector<vector<Connection*>> Node_Path)
{
    vector<double> Errors;
    for (int i = 0; i < Node_Path.size(); i++) {
        double Error = Expected_Outputs[i] - (Nodes[Node_Path[i][Node_Path[i].size() - 1]->dest]->Value * 10);
        Errors.push_back(Error);
        //find the most active connection of this path, and nuge it to the directions of the error vector.
        Connection* Most_Active_Connection = Node_Path[i][Node_Path[i].size() - 1];
        for (auto *j : Node_Path[i]) {
            //calculate the distrance between the error and the current weight.
            if (abs(j->weight - Error) > abs(Most_Active_Connection->weight - Error))
                Most_Active_Connection = j;
        }
        //this most active connection's weight is nudged by the Learning rate.
        Most_Active_Connection->weight = Sigmoid(Most_Active_Connection->weight + Error * Learning_Rate);
    }
    return Errors;
}

Connection::Connection(int src, int dest, double weight)
{
    this->src = src;
    this->dest = dest;
    this->weight = weight;
}

void NN::Clean_Dum_Connections()
{
    for (int i = 0; i < Connections.size(); i++) {
        if (Connections[i]->Dum) {
            Connections.erase(Connections.begin() + i);
        }
    }
}