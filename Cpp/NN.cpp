#include "../H/NN.h"

constexpr double Learning_Rate = 0.0001;

NN::NN(int Height, int Width){
    srand(time(NULL));
    this->Height = Height;
    this->Width = Width;
    for (int i = 0; i < Width * Height; i++){
        Nodes.push_back(new Node());
    }
}

void NN::Save_Weights(string fileName)
{
    std::ofstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        cout << "Cant open file " + fileName << endl;
    }
    else {

        string Buffer = to_string(Inputs_Node_Indices.size()) + "\n";
        for (auto i : Inputs_Node_Indices) {
            Buffer += to_string(i) + "\n";
        }

        Buffer += to_string(Outputs_Node_Indices.size()) + "\n";
        for (auto i : Outputs_Node_Indices) {
            Buffer += to_string(i) + "\n";
        }
            
        Buffer += to_string(Connections.size()) + "\n";
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
    std::ifstream file(fileName, std::ios::binary);
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

            int Input_Node_Count = atoi(Elements[0].c_str());
            int Output_Node_Count = atoi(Elements[Input_Node_Count+1].c_str());

            for (int i = 0; i < Input_Node_Count; i++) {
                Inputs_Node_Indices.push_back(atoi(Elements[i+1].c_str()));
            }

            for (int i = 0; i < Output_Node_Count; i++) {
                Outputs_Node_Indices.push_back(atoi(Elements[i+1 + Input_Node_Count+1].c_str()));
            }

            const int Header = Input_Node_Count + 1 + Output_Node_Count + 1;

            //the first element is the size of the weights.
            Connections = vector<Connection*>(atoi(Elements[Header].c_str()));
            for (auto& i : Connections) {
                i = new Connection();
            }

            for (int i = Input_Node_Count + 1 + Output_Node_Count + 1 + 1; i < Elements.size(); i += 3) {
                int Corrected_Index = (i - Header) / 3;
                
                Connections[Corrected_Index]->src = atoi(Elements[i].c_str());
                Connections[Corrected_Index]->dest = atoi(Elements[i + 1].c_str());
                Connections[Corrected_Index]->weight = atof(Elements[i + 2].c_str());

                Nodes[Connections[Corrected_Index]->dest]->Connections.push_back(Connections[Corrected_Index]);
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

void NN::Add_Connection_Random(int dest)
{
    int src;
    double weight;
    do {
        src = rand() % (Height * Width);
    } while (src == dest);
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
    for (auto Input : Inputs_Node_Indices) {
        for (auto Output : Outputs_Node_Indices) {
            vector<Connection*> Single_Path;
            bool No_Connection = true;
            while (No_Connection) {
                for (auto Con : Nodes[Output]->Connections) {
                    for (auto Path : Find_Path(Con, Input)) {
                        Single_Path.push_back(Path);
                    }
                    if (Single_Path.size() == 0) {
                        Add_Connection_Random(Con->src);
                    }
                }
                if (Single_Path.size() == 0) {
                    Add_Connection_Random(Output);
                }
                else {
                    No_Connection = false;
                }
            }
            Node_Path.push_back(Single_Path);
        }
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
vector<Connection*> NN::Find_Path(Connection* c, int Input_Index)
{
    vector<Connection*> Result;

    for (auto i : Trace) {
        if (i == c) {
            return Result;
        }
    }

    //Check if we have hit the end
    if (c->src == Input_Index) {
        Result.push_back(c);
        return Result;
    }

    vector<Connection*> Connections = Nodes[c->src]->Connections;

    Trace.push_back(c);
    for (auto* i : Connections) {
        if (i->Dum)
            continue;

        int Self_Distance = abs(Input_Index - c->src);
        int Connection_Distance = abs(Input_Index - i->src);

        if (Connection_Distance < Self_Distance) {
            for (auto k : Find_Path(i, Input_Index)) {
                Result.push_back(k);
            }
        }
        else {
            i->Dum = true;
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
    //Calculate first the output node layer errors
    for (auto output : Node_Path)
        for (auto Current_Expected_Output : Expected_Outputs) {
            output[output.size() - 1]->Error = Current_Expected_Output - (Nodes[output[output.size() - 1]->dest]->Value * 10);
            Errors.push_back(output[output.size() - 1]->Error);
        }

    //we dont need to collect all of em.
    //just add one at a time.
    for (int Current_Path = 0; Current_Path < Node_Path.size(); Current_Path++) {
        for (int i = Node_Path[Current_Path].size() - 2; i > 0; i--) {
            Connection* Current = Node_Path[Current_Path][i];
            Connection* Previus = Node_Path[Current_Path][i + 1];

            Current->Error += Previus->weight * Previus->Error;

            Current->Already_Summed.push_back(Previus);
        }
    }
    
    //Run it back, this is to insure that all connections that are cross referenced in different paths are summed up.
    for (int Current_Path = 0; Current_Path < Node_Path.size(); Current_Path++) {
        for (int i = Node_Path[Current_Path].size() - 2; i > 0; i--) {
            Connection* Current = Node_Path[Current_Path][i];
            Connection* Previus = Node_Path[Current_Path][i + 1];

            bool Already_Summed = false;
            for (auto &j : Current->Already_Summed) {
                if (j == Previus) {
                    Already_Summed = true;
                    break;
                }
            }
            if (Already_Summed)
                break;

            Current->Error += Previus->weight * Previus->Error;
        }
    }

    //find the biggest weight on each path :D
    vector<Connection*> Largest_Weights;
    for (auto &Path : Node_Path) {
        Connection* Largest_Connection = Path[0];
        for (auto *C : Path) {
            if (abs(Largest_Connection->weight - Largest_Connection->Error) < abs(C->weight - C->Error)) {
                Largest_Connection = C;
            }
        }
        Largest_Weights.push_back(Largest_Connection);
    }

    //Now that we know the largest weights for every path, we can calculate the nuging.
    for (auto *i : Largest_Weights) {
        i->weight = i->weight + Sigmoid(i->Error) * Learning_Rate;
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
    Update_Connections();
}