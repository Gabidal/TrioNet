#include "../../H/Natural_Selector/Natural_Selector.h"

Natural_Selector::Natural_Selector(Task task, int Amount) : TASK(task)
{
	//init Nets && Reuslts
	for (int i = 0; i < Amount; i++) {
		NN nn(TASK.Parameter_Amount, 1, 1, TASK.Output_Amount);
		Result_Of_NN.push_back(0);
		Nets.push_back(nn);
	}
}

Natural_Selector::Natural_Selector(Task task, int Amount, string Starting_FileName) : TASK(task), Loading_File_Name(Starting_FileName)
{	//init Nets && Reuslts
	for (int i = 0; i < Amount; i++) {
		NN nn(TASK.Parameter_Amount, 1, 1, TASK.Output_Amount);
		Result_Of_NN.push_back(0);
		Nets.push_back(nn);
	}
}

void Natural_Selector::Factory()
{
	for (int k = 0; k < 100000; k++) {
		//load the NN as previus
		for (NN& i : Nets)
			i.Load(Loading_File_Name);
		//mutate all of em besides index 0
		for (int i = 1; i < Nets.size(); i++) {
			Nets.at(i).Mutate();
		}
		for (int i = 0; i < Nets.size(); i++) {
			Trainer.push_back(thread(&Train, this, &Nets.at(i), i));
		}
		while (Trainer.size() > 0)
			for (thread& i : Trainer)
			{
				// If thread Object is Joinable then Join that thread.
				if (i.joinable())
					i.join();
			}

		//determine the lowest cost after the training
		int Smallest = 0;
		int Index = 0;
		for (int i = 0; i < Result_Of_NN.size(); i++)
			if (Result_Of_NN.at(i) < Smallest) {
				Smallest = Result_Of_NN.at(i);
				Index = i;
			}
		//use the lowest costed NN to be loaded next time
		Nets.at(Index).Load(Loading_File_Name);
		//clear the treadhs
		Trainer.clear();
	}
}

Natural_Selector::~Natural_Selector()
{
}

void Natural_Selector::Train(NN* Net, int i)
{
	for (int j = 0; j < 1000000; j++)
		Net->Train();

	vector<double> In;
	Append(In, TASK.Input, TASK.Parameter_Amount);
	vector<double> Out;
	Append(Out, TASK.Expected, TASK.Output_Amount);

	Net->Test(In);

	double Expected = accumulate(TASK.Expected.begin(), TASK.Expected.end(), 0.0);
	double Output = accumulate(Net->Layers.back()->Nodes.begin(), Net->Layers.back()->Nodes.end(), 0.0);

	Result_Of_NN.at(i) = Expected - Output;
}

void Natural_Selector::Append(vector<double>& dest, vector<double> source, int i)
{
	for (int j = 0; j < i; j++)
		dest.push_back(source.at(j));
}