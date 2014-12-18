#include <iostream>

using namespace std;


class Net {
public:
    Net(topology);
    void feedForward(inputVals);
    void backPropagation(targetVals);
    void getResults(resultVals) const;
private:

};


int main() {
    //create neural net
    Net net(topology);
    
    //train neural net
    net.feedForward(inputVals);
    net.backPropagation(targetVals);
    
    //test neural net
    net.getResults(resultVals);

}
