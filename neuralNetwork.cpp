#include "net.h"
#include "neuron.h"
#include "trainingData.h"

void printVector(string label, vector<double>& inVector) {
    cout << label << " ";
    for(int i = 0; i < inVector.size(); i++)
        cout << inVector[i] << " ";
    cout << endl;
}

int main() {

    TrainingData trainingData("trainingData.txt");

    //create neural net
    vector<unsigned int> topology; //ex: (3, 2, 1) - 3 input, 2 hidden, 1 output
    trainingData.getTopology(topology);
    
    Net net(topology);
    
    vector<double> inputVals, targetVals, resultVals;
    int trainingIteration = 0;

    while(!trainingData.isEoF()) {

        //Get input data and feed it forward
        if(trainingData.getNextInputs(inputVals) != topology[0])
            break;

        cout << endl << "Pass " << trainingIteration++;

        printVector(": Inputs:", inputVals);
        net.feedForward(inputVals);

        //Collect net's output results
        net.getResults(resultVals);
        printVector("Outputs:", resultVals);

        //Train the net to improve output
        trainingData.getTargetOutputs(targetVals);
        printVector("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        net.backPropagation(targetVals);

        //how well is training working (averaged over recent samples)?
        cout << "Net recent average error: " << net.getRecentAverageError() << endl;
    }
    cout << endl << "Done training using " << trainingIteration - 1 << " data points!!" << endl << endl;
    cout << "Enter 'q' to exit the program." << endl;

    char input1, input2;
    do {
        inputVals.clear();
        cout << "Enter first input (0 or 1): ";
        cin >> input1;
        cout << "Enter second input (0 or 1): ";
        cin >> input2;

        // '0' -> 0, '1' -> 1
        int in1 = input1 - '0';
        int in2 = input2 - '0';

        inputVals.push_back(in1); 
        inputVals.push_back(in2);

        net.feedForward(inputVals);
        net.getResults(resultVals);

        cout << "Predicted output: " << abs(round(resultVals.front())) << " (actual value: " << resultVals.front() << ")" << endl;
        cout << "Correct output: " << (in1 ^ in2) << endl << endl;

    }while (input1 != 'q' && input2 != 'q');







}