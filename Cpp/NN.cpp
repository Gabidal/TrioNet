#include "../H/NN.h"

constexpr double Learning_Rate = 0.01;

NN::NN(int Height, int Width){
    srand(time(NULL));
    this->Height = Height;
    this->Width = Width;
    for (int i = 0; i < Width * Height; i++){
        Nodes.push_back(new Node());
    }
}

void NN::Set(NN& Og)
{
    Lowest_Error = Og.Lowest_Error;

    for (int i = 0; i < Og.Connections.size(); i++) {
        Connections[i]->Already_Summed.clear();
        Connections[i]->weight = Og.Connections[i]->weight;
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
    //do{
        src = rand() % (Height * Width);
        dest = rand() % (Height * Width);
    //}while(src == dest);
    weight = ((double)rand() / (RAND_MAX)) * 2 - 1;
    Add_Connection(src, dest, weight);
}

void NN::Add_Connection_Random(int dest)
{
    int src;
    double weight;
    //do {
        src = rand() % (Height * Width);
    //} while (src == dest);
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
            int index = rand() % (Height * Width);

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
            int index = rand() % (Height * Width);

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

vector<double> NN::Feed_Forward(vector<double> Inputs, vector<vector<Connection*>> Node_Path)
{
    //Set the input nodes to the input values
    for (int i = 0; i < Inputs.size(); i++)
    {
        Nodes[Inputs_Node_Indices[i]]->Value = Inputs[i];
    }
    //Feed forward the information from the connections a node has.
    for (auto &Path : Node_Path) {
        for (auto& Con : Path) {
            Nodes[Con->dest]->Feed_Forward(this);
        }
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

void NN::Train(int Training_Data_Start_Index, int Training_Data_End_Index, vector<pair<vector<double>, vector<double>>>* Training_Data, double& Error, int Iterations) {
    for (int i = 0; i < Iterations; i++) {
        for (int j = Training_Data_Start_Index; j < Training_Data_End_Index; j++)
        {
            double Avg = 0;
            Feed_Forward(Training_Data->at(j).first, Node_Path);
            vector<double> Errors = Back_Propagation(Training_Data->at(j).second, Node_Path);
            for (auto e : Errors)
                Avg += e;
            Avg /= Errors.size();
            Error += Avg;
        }
        Error /= Iterations;
    }
}

void NN::Start_Train(vector<pair<vector<double>, vector<double>>> Training_Data, int Iterations)
{
    int Core_Count = thread::hardware_concurrency() / 2;
    Update_Path();
    Clean_Dum_Connections();

    vector<NN*> Students;
    vector<double> Errors;
    for (int i = 0; i < Core_Count; i++) {
        Students.push_back(new NN(*this));
        Errors.push_back(0);
    }
    
    vector<pair<vector<double>, vector<double>>>* Data = &Training_Data;
    pair<time_t, double> Iteration_Info = {time(NULL), 100};

    bool Stop = false;
    while (Lowest_Error > 0.01 && Stop == false)
    {
        double All_Avg = 0;

        vector<thread> Threads;

        //update all the students.
            for (auto& S : Students) {
                S->Set(*this);
            }

        for (int T = 0; T < Core_Count; T++) {
            //cout << T * (Data->size() / Core_Count) << " -> " << (T + 1) * (Data->size() / Core_Count) << endl;
            Threads.push_back(thread([T, &Student = Students[T], Data, Core_Count, &Error = Errors[T], Iterations]() {
                Student->Train(T* (Data->size() / Core_Count), (T + 1)* (Data->size() / Core_Count), Data, Error, Iterations);
            }));
        }

        for (int T = 0; T < Threads.size(); T++) {
            Threads[T].join();

            All_Avg += Errors[T];
        }

        //clean first the connection weight values from the og
        for (auto* C : Connections) {
            C->weight = 0;
            C->Error = 0;
        }

        //Combine the connection weights
        for (auto* nn : Students) {
            for (int C = 0; C < min(Connections.size(), nn->Connections.size()); C++) {
                Connections[C]->weight += nn->Connections[C]->weight;
                Connections[C]->Error += nn->Connections[C]->Error;
            }
        }

        for (auto* C : Connections) {
            C->weight /= Students.size();
            C->Error /= Students.size();
        }

        Lowest_Error = All_Avg / Training_Data.size();
        cout << "Error: " + to_string(All_Avg / Training_Data.size()) << endl;

        if (Iteration_Info.second > Lowest_Error) {
            Iteration_Info.first = time(NULL);
            Iteration_Info.second = Lowest_Error;
        }
        else if (abs(Iteration_Info.first - time(NULL)) >= 1000 * 60 * 5) {
            Stop = true;
        }
    }
    Clean_Dum_Connections();
}

vector<Connection*> NN::Find_Path(Connection* c, int Input_Index, vector<Connection*> Trace)
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
            for (auto k : Find_Path(i, Input_Index, Trace)) {
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
            output[output.size() - 1]->Error += -(Current_Expected_Output - (Nodes[output[output.size() - 1]->dest]->Value * 10)) * Sigmoid_Derivative(Nodes[output[output.size() - 1]->dest]->Value);
            Errors.push_back(abs(Current_Expected_Output - (Nodes[output[output.size() - 1]->dest]->Value * 10)));
        }


    //Re order the node_path
    vector<vector<Connection*>> Re_Ordered_Node_Path;
    Re_Ordered_Node_Path = Reorder_Connections(Node_Path);
    //First go through the Node_Path list and take one connection at a time
    //and calculate the error for the connection by summing all previous connection errors.
    //skip the first index because its already calculated.
    for (int i = 1; i < Re_Ordered_Node_Path.size(); i++) {
        for (int Current = 0; Current < Re_Ordered_Node_Path[i].size(); Current++) {
            //clear the error to prevent escalation.
            double Error = 0;
            //Sum all the errors from the previous connections
            for (int Prev = 0; Prev < Re_Ordered_Node_Path[i - 1].size(); Prev++) {
                Error += Re_Ordered_Node_Path[i - 1][Prev]->Error * Re_Ordered_Node_Path[i - 1][Prev]->weight;
            }
            //Calculate the error for the current connection
            Re_Ordered_Node_Path[i][Current]->Error = Error * Sigmoid_Derivative(Nodes[Re_Ordered_Node_Path[i][Current]->src]->Value);
        }
    }

    for (auto& Path : Node_Path) {
        for (auto* C : Path) {
            C->Already_Summed.clear();
        }
    }

    //find the biggest weight on each path :D
    vector<Connection*> Largest_Weights;
    for (auto &Path : Node_Path) {
        Connection* Largest_Connection = Path[0];
        for (auto *C : Path) {
            C->Error = C->Error * Sigmoid_Derivative(Nodes[C->dest]->Value);

            if (abs(Largest_Connection->weight - Largest_Connection->Error) < abs(C->weight - C->Error)) {
                Largest_Connection = C;
            }
        }
        Largest_Weights.push_back(Largest_Connection);
    }

    //Now that we know the largest weights for every path, we can calculate the nuging.
    for (auto& Path : Largest_Weights)
        for (auto* i : Largest_Weights) {
            i->weight -= i->Error * Learning_Rate * Nodes[i->src]->Value;
        }

    ////clean the errors
    //for (auto& Path : Node_Path)
    //    for (auto& C : Path)
    //        C->Error = 0;

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
    for (int i = Connections.size() - 1; i >= 0; i--) {
        Connections[i]->Dum = true;
        for (auto& Path : Node_Path) {
            for (auto& Con : Path) {
                if (Connections[i] == Con) {
                    Connections[i]->Dum = false;
                }
            }
        }        
        if (Connections[i]->Dum) {
            delete Connections[i];
            Connections.erase(Connections.begin() + i);
        }
    }
    Update_Connections();
}

NN::NN(const NN& Og)
{
    Lowest_Error = Og.Lowest_Error;
    Height = Og.Height;
    Width = Og.Width;

    for (auto &i : Og.Connections) {
        i->Already_Summed.clear();
        Connections.push_back(new Connection(*i));
    }
    for (auto &i : Og.Nodes) {
        i->Connections.clear();
        Nodes.push_back(new Node(*i));
    }

    Node_Path.clear();
    Inputs_Node_Indices = Og.Inputs_Node_Indices;
    Outputs_Node_Indices = Og.Outputs_Node_Indices;

    Update_Connections();

    Update_Path();
    Clean_Dum_Connections();
}

void NN::Update_Path()
{
    //Try to path find through the connections, a conneciton that starts from a certain input node, and end in a certain output node.
    //Gather these connections into a list, this list will be back propagated.
    for (auto& Input : Inputs_Node_Indices) {
        for (auto& Output : Outputs_Node_Indices) {
            vector<Connection*> Single_Path;
            bool No_Connection = true;
            while (No_Connection) {
                vector<Connection*> Path_Connections = Nodes[Output]->Connections;
                for (auto &Con : Path_Connections) {
                    for (auto& Path : Find_Path(Con, Input, vector<Connection*>())) {
                        Single_Path.push_back(Path);
                    }
                    if (Single_Path.size() == 0) {
                        Add_Connection_Random(/*Con->src*/);
                        for (auto& Path : Find_Path(Con, Input, vector<Connection*>())) {
                            Single_Path.push_back(Path);
                        }
                    }
                    if (Single_Path.size() != 0)
                        break;
                }
                if (Single_Path.size() > 0) {
                    No_Connection = false;
                }
                else {
                    Add_Connection_Random(/*Output*/);
                }
            }
            Node_Path.push_back(Single_Path);
        }
    }
}

vector<vector<Connection*>> NN::Reorder_Connections(vector<vector<Connection*>> Node_Path)
{
    //First connect all different Node_Paths if they have same output node.
    vector<vector<Connection*>> New_Node_Path;
    for (auto& Path : Node_Path) {
        bool Found = false;
        for (auto& New_Path : New_Node_Path) {
            if (New_Path[New_Path.size() - 1]->dest == Path[Path.size() - 1]->dest) {
                Found = true;
                New_Path.insert(New_Path.end(), Path.begin(), Path.end());
            }
        }
        if (!Found) {
            New_Node_Path.push_back(Path);
        }
    }

    //Make a new list that constains all connection distances from the output node.
    vector<pair<int, vector<Connection*>>> Connection_Distances;
    for (auto& Path : New_Node_Path) {
        for (auto& Con : Path) {
            //try to find if this distance has already been achieved
            bool Found = false;
            for (auto& Previus : Connection_Distances) {
                if (Previus.first == abs(Con->dest - Path[Path.size() - 1]->dest)) {
                    Previus.second.push_back(Con);
                    Found = true;
                }
            }
            if (!Found)
                Connection_Distances.push_back({ abs(Con->dest - Path[Path.size() - 1]->dest), {Con } });
        }
    }

    //Sort the list by the distance from the output node.
    sort(Connection_Distances.begin(), Connection_Distances.end());

    //Now construct the new Node_Path, by adding the connections in the correct order.
    vector<vector<Connection*>> New_Node_Path2;
    for (auto& Path : Connection_Distances) {
        New_Node_Path2.push_back(Path.second);
    }
    
    return New_Node_Path2;
}