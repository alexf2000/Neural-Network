#ifndef NET
#define NET

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

class Neuron; //forward reference - Layer and Neuron need each other
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

class Net {
public:
    Net(const vector<unsigned int>& topology);
    void feedForward(const vector<double>& inputVals);
    void backPropagation(const vector<double>& targetVals);
    void getResults(vector<double>& resultVals) const;
    double getRecentAverageError() const { return mRecentAverageError; }
private:
    vector<Layer> mLayers; //[layerNumber][neuronNumber]
    double mError, mRecentAverageError, mRecentAverageSmoothingFactor;
};

#endif