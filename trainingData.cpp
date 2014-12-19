#include "trainingData.h"

TrainingData::TrainingData(const string filename) {
    mDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned int>& topology) {
    string line, label;

    getline(mDataFile, line);
    stringstream stream(line);
    stream >> label;
    
    if(this->isEoF() || label.compare("topology:") != 0)
        abort();

    int tmp;
    while(!stream.eof()) {
        stream >> tmp;
        topology.push_back(tmp);
    }
}

int TrainingData::getNextInputs(vector<double>& inputVals) {
    inputVals.clear();

    string line, label;
    getline(mDataFile, line);
    stringstream stream(line);
    stream >> label;

    if(label.compare("in:") == 0) {
        double tmp;
        while(stream >> tmp)
            inputVals.push_back(tmp);
    }

    return inputVals.size();
}

int TrainingData::getTargetOutputs(vector<double>& targetVals) {
    targetVals.clear();

    string line, label;
    getline(mDataFile, line);
    stringstream stream(line);
    stream >> label;

    if(label.compare("out:") == 0) {
        double tmp;
        while(stream >> tmp)
            targetVals.push_back(tmp);
    }

    return targetVals.size();
}