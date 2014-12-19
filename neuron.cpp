#include "neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

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
        sum += mOutputWeights[i].weight * nextLayer[i].mGradient;
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