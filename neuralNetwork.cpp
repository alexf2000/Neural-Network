#include <iostream>
#include <vector>

using namespace std;


class Neuron {
};

typedef vector<Neuron> Layer;

class Net {
public:
    Net(const vector<unsigned int>& topology);
    void feedForward(const vector<double>& inputVals);
    void backPropagation(const vector<double>& targetVals);
    void getResults(vector<double>resultVals) const;
private:
    vector<Layer> mLayers; //[layerNumber][neuronNumber]

};

Net::Net(const vector<unsigned int>& topology) {
    int numLayers = topology.size();
   
    //add layers to mLayers
    for(int i = 0; i < numLayers; i++) {
        mLayers.push_back(Layer());
     
        //add topology[i] number of neurons (+1 bias neuron) to new layer
        for(int j = 0; j < topology[i] + 1; j++) {
            mLayers.back().push_back(Neuron());
            cout << "Made a neuron!" << endl;
        }
    };
}



void Net::feedForward(const vector<double>& inputVals) {

}
void Net::backPropagation(const vector<double>& targetVals) {

}
void Net::getResults(vector<double>resultVals) const {

}


int main() {
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
