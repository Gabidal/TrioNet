#include <string>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

//One connection can connect up to two nodes.
//The connection specifies the src and dest nodes by indicees.
class Connection{
    public:
        int src;
        int dest;
        double weight;
        double deltaWeight;
        double error;

        Connection(int src, int dest, double weight);

        
};

//imagine a 2D grid of nodes.
//In this grid the nodes are handled by indices.
//The indices are used to access the nodes in the grid.
//The connections work by these indices. 
class NN{
    public:
        double* Nodes;
        //To prevent data copying use pointers.
        vector<Connection*> Connections;
        //We want to make the output and the intput nodes to be runtime determine.
        //This way the neuralnetwork may be used for different problems.
        vector<int> Inputs_Node_Indices;
        vector<int> Outputs_Node_Indices;
        int Height;
        int Width;

        NN(int Height, int Width);

        //Make a function that can load the weights from a file.
        //The first section tells the src node index, the second section tells the dest node index, and the third section tells the weight.
        //These sections are seperated by commad
        void Save_Weights(string fileName);

        //Make a function that can load the weights from a file.
        //The first section tells the src node index, the second section tells the dest node index, and the third section tells the weight.
        //These sections are seperated by commad
        void Load_Weights(string fileName);

        //This function is used to add a connection to the neural network.
        //The connection is added to the vector of connections.
        void Add_Connection(int src, int dest, double weight);

        //This function is used to add a connection with random src and dest node indices.
        //Also the weight is set to a random value between -1 and 1.
        //The connection is added to the vector of connections.
        void Add_Connection_Random();


        //This function is the activation function.
        //We use the sigmoid function to activate the nodes.
        //The sigmoid function is defined as 1/(1+e^-x)
        //The function returns the activated value.
        double Sigmoid(double x);

        //This function detects if the input vector is larger than the input node indices.
        //If true then this function resizes the input vector to the size of the input node indices.
        void Resize_Input_Vector(int input);

        //This function detects if the output vector is larger than the output node indices.
        //If true then this function resizes the output vector to the size of the output node indices.
        void Resize_Output_Vector(int output);

        //This function does the feed forward calculation.
        //The input is the input vector.
        //The output is gathered when a node is an output node. (this can be checked by the output node indices)
        //We also need to make sure that the Input vector given to this function is same size as the input nodes.
        //If not then use the resize function to resize the input vector.
        vector<double> Feed_Forward(vector<double> Input);

        //This function can get a function pointter that is used to generate the training data set.
        //The function pointer gets a list of inputs and then return a list of outputs.
        vector<pair<vector<double>, vector<double>>> Get_Training_Data(vector<double> (*Generate_Training_Data)(vector<double>), int in, int out);

        //This function will train the neural network besed on the training data set that we just generated.
        //The training data set is a list of inputs and outputs.
        void Train(vector<pair<vector<double>, vector<double>>> Data, int epochs);

        //This function does the backpropagation calculation.
        //The input is the output vector.
        //The output is the error vector.
        //We also need to make sure that the Output vector given to this function is same size as the output nodes.
        //If not then use the resize function to resize the output vector.
        vector<double> Back_Propagation(vector<double> Output);

        //This function goes backwards through a given list of connections and returns the next list of connections.
        //The input is the list of connections.
        //The output is the list of connections.
        vector<Connection*> Get_Previus_Connections(vector<Connection*> Connections);

};
