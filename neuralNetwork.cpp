#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;


class Neuron {}; //forward reference - Layer and Neuron need each other

typedef vector<Neuron> Layer;

//for connections between neurons
struct Connection {
    double weight;
    double deltaWeight; 

    Connection() {
        //initialize weight to random number 0 < weight < 1
        weight = rand() / double(RAND_MAX);
    }
};

// --------------------- class TrainingData ----------------------------
class TrainingData {
public:
    TrainingData(const string filename);
    bool isEoF(void) { return mDataFile.eof(); }
    void getTopology(vector<int>& topology);
    int getNextInputs(vector<double>& inputVals);
    int getTargetOutputs(vector<double>& targetOutputVals);
private:
    ifstream mDataFile;
};

void TrainingData::TrainingData(const string filename) {
    mDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned> &topology) {
    string line, label;

    getline(mDataFile, line);
    stringstream stream(line);
    stream >> label;
    
    if(this->isEoF || label.compare("topology:") != 0)
        abort();

    int tmp;
    while(!stream.eof()) {
        ss >> tmp;
        topology.push_back(tmp);
    }
}

int TrainingData::getNextInputs(vector<double> &inputVals) {
    inputVals.clear();

    string line, label;
    getline(mDataFile, line);
    stringstream stream(line);
    ss >> label;

    if(label.compare("in:") == 0) {
        double tmp;
        while(ss >> tmp)
            inputVals.push_back(tmp);
    }

    return inputVals.size();
}

int TrainingData::getTargetOutputs(vector<double> &targetVals) {
    inputVals.clear();

    string line, label;
    getline(mDataFile, line);
    stringstream stream(line);
    ss >> label;

    if(label.compare("out:") == 0) {
        double tmp;
        while(ss >> tmp)
            targetVals.push_back(tmp);
    }

    return targetVals.size();
}



// ------------------------- class Neuron ------------------------------
class Neuron {
public:
    Neuron(int numOutputs, int index);
    void setOutputVal(double val) { mOutputVal = val };
    double getOutputVal() const { return mOutputVal };
    void feedForward(const Layer& prevLayer);
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    double sumDOW(const Layer& nextLayer);
    double mOutputVal, mGradient;
    vector<Connection> mOutputWeights;
    int mIndex;
    static double eta = 0.15; //overall net learning rate
    static double alpha = 0.5; //momentum, multiplier of last deltaWeight
};

Neuron::Neuron(int numOutputs, int index) {
    for(int i = 0; i < numOutputs; i++) {
        mOutputWeights.push_back(Connection());
    }
    mIndex = index;
}

void Neuron::feedForward(const Layer& prevLayer) {
    double sum = 0;

    //sum previous layer's outputs and bias neuron (this current neuron's inputs)
    for(int i = 0; i < prevLayer.size(); i++) {
        sum += prevLayer[i].getOutputVal() * 
                prevLayer[i].mOutputWeights[mIndex].weight;
    }

    mOutputVal = Neuron::activationFunction(sum);
}

double Neuron::activationFunction(double x) {
    //need derivative, so use hyperbolic tangent function
    //tanhx = (e^x - e^-x)/(e^x  + e^-x)    -- between -1 and 1

    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    //derivative of tanhx = 1 - tanh^2(x)
    //use quick approximation - 1 - x^2

    return 1.0 - x * x;
}

void Neuron::calculateOutputGradients(double targetVal) {
    double difference = targetVal - mOutputVal;
    mGradient = difference * Neuron::activationFunctionDerivative(mOutputVal);
}

void Neuron::calculateHiddenGradients(const Layer& nextLayer) {
    double dow = sumDOW(nextLayer); //sum of derivatives of next layer
    mGradient = dow * Neuron::activationFunctionDerivative(mOutputVal);
}

double Neuron::sumDOW(const Layer& nextLayer) {
    double sum = 0;

    for(int i = 0; i < nextLayer.size() - 1; i++) {
        sum += mOutputWeight[i].weight * nextLayer[i].mGradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer& prevLayer) {
    //Weights to be updated are in Connection container in neurons in preceding layer
    
    for(int i = 0; i < prevLayer.size(); i++) {
        Neuron& neuron = prevLayer[i];
        double oldDeltaWeight = neuron.mOutputWeights[mIndex].deltaWeight;
    
        //eta - 0.0 (slow), 0.2 (medium), 1.0 (reckless)
        double newDeltaWeight = 
            //individual input magnified by gradient and training rate:
            eta * neuron.getOutputVal() * mGradient
            //add momentum = a fraction of the previous delta weight
            + alpha * oldDeltaWeight;
       
       neuron.mOutputWeights[mIndex].deltaWeight = newDeltaWeight;
       neuron.mOutputWeights[mIndex].weight += newDeltaWeight;
    
    }

}
// ------------------------- class Net --------------------------------
class Net {
public:
    Net(const vector<unsigned int>& topology);
    void feedForward(const vector<double>& inputVals);
    void backPropagation(const vector<double>& targetVals);
    void getResults(vector<double>resultVals) const;
private:
    vector<Layer> mLayers; //[layerNumber][neuronNumber]
    double mError, mRecentAverageError, mRecentAverageSmoothingError;
};

Net::Net(const vector<unsigned int>& topology) {
    int numLayers = topology.size();
   
    //add layers to mLayers
    for(int i = 0; i < numLayers; i++) {
        mLayers.push_back(Layer());
        int numOutputs = i == numLayers - 1 ? 0 : topology[i + 1]; //output node has 0 outputs

        //add topology[i] number of neurons (+1 bias neuron) to new layer
        for(int j = 0; j < topology[i] + 1; j++) {
            mLayers.back().push_back(Neuron(numOutputs));
            cout << "Made a neuron!" << endl;
        }

        //Force bias neuron's output to be 1.0
        mLayers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double>& inputVals) {
    assert(inputVals.size() == mLayers[0].size() - 1); //is #inputVals == #input neurons?
    
    //Assign the input values to input neurons
    for(int i = 0; i < inputVals.size(); i++) {
        mLayers[0][i].setOutputVal(inputVals[i]);
    }

    //forward propagation - tell each neuron in next layers to feed forward
    for(int i = 1; i < mLayers.size(); i++) {
        Layer &prevLayer = mLayers[i - 1];
        for(int j = 0; j < mLayers[i].size() - 1; j++) { // -1 because of bias neuron
            mLayers[i][j].feedForward(prevLayer); //updates output val
        }
    }
}

void Net::backPropagation(const vector<double>& targetVals) {
    //calculate overall net error (Root mean square of output neuron errors)
    Layer &outputLayer = mLayers.back();
    mError  = 0.0;

    for(int i = 0; i < outputLayer.size() - 1; i++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        mError += delta * delta;
    }
    mError /= outputLayer.size() - 1;
    mError = sqrt(mError);

    //implement a recent average measurement
    mRecentAverageError = (mRecentAverageError * mRecentAverageSmoothingFactor + mError)
        / (mRecentAverageSmoothingFactor + 1.0);


    //calculate output layer gradients
    for(int i = 0; i < outputLayer.size() - 1; i++) {
        outputLayer[i].calculateOutputGradients(targetVal[i]);
    }
    
    //calculate gradients on hidden layers
    for(int i = mLayers.size() - 2; i > 0; i--) {
        Layer &hiddenLayer = mLayers[i]; //current layer
        Layer &nextLayer = mLayers[i + 1];

        for(int j = 0; j < hiddenLayer.size(); j++) {
            hiddenLayer[j].calculateHiddenGradients(nextLayer);
        }
    }

    //update connection weights from outputs backwards to first hidden layer
    for(int i = mLayers.size() - 1; i > 0; i++) {
        Layer &curLayer = mLayers[i];
        Layer &prevLayer = mLayers[i - 1];

        for(int j = 0; j < curLayer.size() - 1; j++) {
            curLayer[j].updateInputWeights(prevLayer);
        }
    }
}
void Net::getResults(vector<double>resultVals) const {
    resultVals.clear();

    for(int i = 0; i < mLayers.back().size() - 1; i++) {
        resultVals.push_back(mLayers.back()[i].getOutputVal());
    }
}


int main() {

    TrainingData trainingData("trainingData.txt")


    //create neural net
    vector<unsigned int> topology; //ex: (3, 2, 1) - 3 input, 2 hidden, 1 output
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    
    Net net(topology);
    
    
    vector<double> inputVals, targetVals, resultVals;

    //train neural net
    net.feedForward(inputVals);
    net.backPropagation(targetVals);
    
    //test neural net
    net.getResults(resultVals);

}
