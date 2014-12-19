#ifndef NEURON
#define NEURON

#include "net.h"

class Neuron {
public:
    Neuron(int numOutputs, int index);
    void setOutputVal(double val) { mOutputVal = val; }
    double getOutputVal() const { return mOutputVal; }
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
    static double eta; //overall net learning rate
    static double alpha; //momentum, multiplier of last deltaWeight
};

#endif