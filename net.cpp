#include "net.h"
#include "neuron.h"

Net::Net(const vector<unsigned int>& topology) {
    int numLayers = topology.size();
   
    //add layers to mLayers
    for(int i = 0; i < numLayers; i++) {
        mLayers.push_back(Layer());
        int numOutputs = i == numLayers - 1 ? 0 : topology[i + 1]; //output node has 0 outputs

        //add topology[i] number of neurons (+1 bias neuron) to new layer
        for(int j = 0; j < topology[i] + 1; j++) {
            mLayers.back().push_back(Neuron(numOutputs, j));
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
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        mError += delta * delta;
    }
    mError /= outputLayer.size() - 1;
    mError = sqrt(mError);

    //implement a recent average measurement
    mRecentAverageError = (mRecentAverageError * mRecentAverageSmoothingFactor + mError)
        / (mRecentAverageSmoothingFactor + 1.0);

    //calculate output layer gradients
    for(int i = 0; i < outputLayer.size() - 1; i++) {
        outputLayer[i].calculateOutputGradients(targetVals[i]);
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
    for(int i = mLayers.size() - 1; i > 0; i--) {
        Layer &curLayer = mLayers[i];
        Layer &prevLayer = mLayers[i - 1];

        for(int j = 0; j < curLayer.size() - 1; j++) {
            curLayer[j].updateInputWeights(prevLayer);
        }
    }
}
void Net::getResults(vector<double>& resultVals) const {
    resultVals.clear();

    for(int i = 0; i < mLayers.back().size() - 1; i++) {
        resultVals.push_back(mLayers.back()[i].getOutputVal());
    }
}
