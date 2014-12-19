all: neuralNetwork.o net.o neuron.o trainingData.o
		g++ neuralNetwork.o net.o neuron.o trainingData.o -o neuralNet
neuralNetwork.o: neuralNetwork.cpp
		@g++ -c neuralNetwork.cpp
net.o: net.cpp net.h
		@g++ -c net.cpp
neuron.o: neuron.cpp neuron.h
		@g++ -c neuron.cpp
trainingData.o: trainingData.cpp trainingData.h
		@g++ -c trainingData.cpp

clean:
		@rm -f *.o neuralNet
