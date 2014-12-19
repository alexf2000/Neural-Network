#ifndef TRAININGDATA
#define TRAININGDATA

#include <fstream>
#include <vector>
#include <sstream>

using namespace std;


class TrainingData {
public:
    TrainingData(const string filename);
    bool isEoF() { return mDataFile.eof(); }
    void getTopology(vector<unsigned int>& topology);
    int getNextInputs(vector<double>& inputVals);
    int getTargetOutputs(vector<double>& targetOutputVals);
private:
    ifstream mDataFile;
};

#endif