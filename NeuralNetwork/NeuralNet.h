//---------------------------------------------------------------------------
#ifndef NEURALNET_H
#define NEURALNET_H
//---------------------------------------------------------------------------

#include <vector>
#include <functional>


namespace CS397
{

using NetworkWeights = std::vector<std::vector<std::vector<double>>>;

struct NNLayerDesc
{
    unsigned                 numNeurons;
    ActivationFunction::Type functionType = ActivationFunction::Type::eSigmoid;
};

class NeuralNet
{
public:
    // constructor where the data to train the neural network is passed
    // it will initialize every neuron with a random weight on the range [-1,1]
	NeuralNet(  const DatasetCreator::Dataset & data,       // dataset to train the network
                const std::vector<NNLayerDesc> & topology,  // number of layers and amount of neurons in each layer and activation function
                double lr                                   // learning rate
                );

    // given an input value, it will propagate those values through the network and produce an output
    // the prediction function of the neural network
	std::vector<double> ForwardPropagation(const std::vector<double> & input);
    
    // produces a train interation with the whole dataset, all weights will be updated
    // first it will produce an output with the forward propagation and the it will back propagate the error
    // updating the weights
	void Iteration();
	
    // returns every weight in the neural network (mainly to check random initialization
	NetworkWeights GetWeights() const;
    
    // computes the cost of the network for the provided dataset
	double Cost(const DatasetCreator::Dataset & data);


private:
    NetworkWeights                      mWeights;
    NetworkWeights                      mComputedWeights;
    DatasetCreator::Dataset             mDataset;
    std::vector<NNLayerDesc>            mTopology;
    double                              mLearningRate;
    unsigned                            mDataSize;
    std::vector<std::vector<double>>    mComputedOutputs;
    std::vector<std::vector<double>>    mGradients;

    double ComputeValueByActivationFunction(const double& value, ActivationFunction::Type functionType);
    double ComputeDerivativeValueByActivationFunction(const double& value, ActivationFunction::Type functionType);
    void BackPropagate(const unsigned layer_idx, const unsigned neuron_idx);
    void ComputeOutputLayerGradient(const std::vector<double>& computedOutput);
    void ComputeHiddenLayerGradient(const unsigned layer_idx);
};

}
#endif