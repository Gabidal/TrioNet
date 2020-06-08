#include <iostream>
#include "../H/NN/NN.h"
#include <time.h>
#include <fstream>
using namespace std;

vector<double> Fibonakki(vector<double> input)
{
    double n = input.at(0);
    double a, b, c;
    vector<double> result;
    a = 0;
    b = 1;
    if (n == 1) {
        return { 0 };
    }
    if (n == 2) {
        return { 1 };
    }

    // Using for loop for continuing 
    // the Fibonacci series. 
    for (int i = 1; i <= n - 2; i++) {

        // Addition of the previous two terms 
        // to get the next term. 
        c = a + b;

        // Converting the new term 
        // into an old term to get 
        // more new terms in series. 
        a = b;
        b = c;
    }
    result.push_back(c);
    return result;
}

int main() {
	srand(time(0));
	NN n(1, 10, 2, 1);
	//input		//AND
    n.Generate_Data_Set(Fibonakki, 1, 2, 30);
    vector<double> costs;
    n.Load("Saved_Weights.txt");
    for (int i = 0; i < 1000000; i++) {
        double result = n.Train();
        if (i % 10000 == 0) {
            costs.push_back(result);
        }
        if (i % 10000 == 0) {
            cout << result << endl;
        }
    }
    n.Save("Saved_Weights.txt");
    ofstream cost_export("cost_function.csv");
    for (int i = 0; i < costs.size(); i++) {
        cost_export << i << "; " << costs.at(i) << endl;
    }
    cost_export.close();
	n.Test({1});
    cout << "1 gived " << n.Output(0) << endl;
	n.Test({6});
    cout << "6 gived " << n.Output(0) << endl;
	n.Test({4});
    cout << "4 gived " << n.Output(0) << endl;
    n.Test({ 27 });
    cout << "27 gived " << n.Output(0) << endl;
    n.Test({ 31 });
    cout << "31 gived " << n.Output(0) << endl;

	return 0;
}