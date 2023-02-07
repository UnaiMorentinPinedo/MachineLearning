/*-------------------------------------------------------
File Name: NeuralNet.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "NeuralNet.h"
#include "PRNG.h"

namespace CS397 {
/*---------------------------------------------------------------------------*
  Name:         NeuralNet
  Description:  Constructor of the class NeuralNet
 *---------------------------------------------------------------------------*/
	NeuralNet::NeuralNet(const DatasetCreator::Dataset& data, const std::vector<NNLayerDesc>& topology, double lr) 
	: mDataset(data), mTopology(topology), mLearningRate(lr)	{
		//compute the size of the dataset
		mDataSize = data.first.empty() ? 0u : data.first.front().size();

		//initialize neural weights
		for (unsigned i = 0u; i < mTopology.size(); i++) {
			unsigned numberOfNeurons_thisLayer = mTopology[i].numNeurons;

			//handle we are already in the output layer
			if (i == mTopology.size() - 1u)
				continue;
			
			//get next layer number of neurons
			unsigned numberOfNeurons_nextLayer = mTopology[i+1].numNeurons;

			//iterate in this layer neurons
			std::vector<std::vector<double>> layer_weights;
			//+1 in the size is for adding the bias weight
			for (unsigned n = 0u; n < numberOfNeurons_thisLayer + 1; n++) {
				std::vector<double> neurons_weight;
				//iterate in next layer neurons
				for (unsigned k = 0u; k < numberOfNeurons_nextLayer; k++) {
					neurons_weight.push_back(PRNG::RandomNormalizedDouble());
				}

				layer_weights.push_back(neurons_weight);
			}

			mWeights.push_back(layer_weights);
		}

		mComputedWeights = mWeights;
		mGradients.resize(mTopology.size()-1u);
		for (unsigned i = 0u; i < mGradients.size(); i++) {
			mGradients[i].resize(mTopology[i+1].numNeurons + 1);
		}
	}

/*---------------------------------------------------------------------------*
	 Name:         ForwardPropagation
	 Description:  Computes values of each neuron taking into account weights
*---------------------------------------------------------------------------*/
	std::vector<double> NeuralNet::ForwardPropagation(const std::vector<double>& input) {
		//prepare vectos for this iteration usage
		std::vector<double> neurons = input;
		std::vector<double> output;
		mComputedOutputs.clear();

		//we dont compute values for the input layer
		mComputedOutputs.push_back(input);
		
		//iterate thorugh the layers
		for (unsigned layer_idx = 0u; layer_idx < mTopology.size()-1; layer_idx++) {
			//get number of links of each neuron = next layer number of neurons
			for (unsigned link_idx = 0u; link_idx < mTopology[layer_idx + 1].numNeurons; link_idx++) {
				double computedValue = 0.;
				for (unsigned neuron_idx = 0u; neuron_idx < mTopology[layer_idx].numNeurons + 1; neuron_idx++) {
					//get input value
					const double value = neuron_idx < neurons.size() ? neurons[neuron_idx] : 1.0;
					const double& weight = mWeights[layer_idx][neuron_idx][link_idx];

					computedValue += value * weight;
				}

				output.push_back(ComputeValueByActivationFunction(computedValue, mTopology[layer_idx+1].functionType));
			}
			mComputedOutputs.push_back(output);
			//copy output values for next layer computation
			neurons = output;
			output.clear();
		}

		return neurons;
	}

/*---------------------------------------------------------------------------*
  Name:         Iteration
  Description:  Main logic of the program
 *---------------------------------------------------------------------------*/
	void NeuralNet::Iteration() {
		std::vector<double> finalOutput = ForwardPropagation(mDataset.first[0]);

		//start at the output layer and backpropagate to the input layer
		ComputeOutputLayerGradient(finalOutput);

		for (unsigned layer_idx = mTopology.size() - 2u; layer_idx > 0u; layer_idx--)
			ComputeHiddenLayerGradient(layer_idx);

		//start in the last hidden layer
		for (unsigned layer_idx = 1u; layer_idx < mTopology.size(); layer_idx++) {
			for (unsigned neuron_idx = 0u; neuron_idx < mTopology[layer_idx].numNeurons; neuron_idx++) {
				BackPropagate(layer_idx, neuron_idx);
			}
		}

		mWeights = mComputedWeights;
	}

	/*---------------------------------------------------------------------------*
	  Name:         ComputeOutputLayerGradient
	  Description:  Computes the gradient for the last layer
	 *---------------------------------------------------------------------------*/
	void NeuralNet::ComputeOutputLayerGradient(const std::vector<double>& computedOutput) {
		for (unsigned neuron_idx = 0u; neuron_idx < mTopology.back().numNeurons; neuron_idx++) {
			//get dataset output value
			const double& datasetOutputValue = mDataset.second[0][neuron_idx]; 
			//dE/do
			double firstDerivative = computedOutput[neuron_idx] - datasetOutputValue;
			//do/dnet
			double secondDerivative = ComputeDerivativeValueByActivationFunction(computedOutput[neuron_idx],
				mTopology.back().functionType);

			mGradients.back()[neuron_idx] = firstDerivative * secondDerivative;
		}
	}

/*---------------------------------------------------------------------------*
  Name:         ComputeHiddenLayerGradient
  Description:  Computes the gradient for the hidden layers
 *---------------------------------------------------------------------------*/
	void NeuralNet::ComputeHiddenLayerGradient(const unsigned layer_idx) {
		//iterate through the neurons of this layer_idx
		for (unsigned neuron_idx = 0u; neuron_idx < mTopology[layer_idx].numNeurons; neuron_idx++) {
			const double& computedOutput = mComputedOutputs[layer_idx][neuron_idx];

			double thisGradientValue = 0.;
			// if connected to a layer with multiple neurons
			for (unsigned connected_to = 0u; connected_to < mTopology[layer_idx + 1].numNeurons; connected_to++) {
				thisGradientValue += mGradients[layer_idx][connected_to] *
					mWeights[layer_idx][neuron_idx][connected_to];
			}

			//do/dnet
			double secondDerivative = ComputeDerivativeValueByActivationFunction(computedOutput, mTopology[layer_idx].functionType);
			//store value in vector
			mGradients[layer_idx-1][neuron_idx] = thisGradientValue * secondDerivative;
		}
	}

/*---------------------------------------------------------------------------*
  Name:         BackPropagate
  Description:  Updates the weights of each link
 *---------------------------------------------------------------------------*/
	void NeuralNet::BackPropagate(const unsigned layer_idx, const unsigned neuron_idx) {
		//get number of links connecetd to this layer neurons = neurons in previous layer
		const unsigned& prev_layer_num_neurons = mTopology[layer_idx - 1].numNeurons;

		//if we are neurom [0] all previous layers neurons' link [0] will be attached to us
		for (unsigned prev_neuron_idx = 0u; prev_neuron_idx < prev_layer_num_neurons + 1; prev_neuron_idx++) {
			const double& weight = mWeights[layer_idx - 1][prev_neuron_idx][neuron_idx];

			//dnet/dw
			double incomingNeuron_output = 0.;
			prev_neuron_idx == prev_layer_num_neurons ? incomingNeuron_output = 1. :
				incomingNeuron_output = mComputedOutputs[layer_idx - 1][prev_neuron_idx];
			
			//store the new weight 
			mComputedWeights[layer_idx - 1][prev_neuron_idx][neuron_idx] = weight - mLearningRate * 
				(mGradients[layer_idx - 1][neuron_idx] * incomingNeuron_output);
		}
	}

/*---------------------------------------------------------------------------*
  Name:         GetWeights
  Description:  Gettor
 *---------------------------------------------------------------------------*/
	NetworkWeights NeuralNet::GetWeights() const {
		return mWeights;
	}

/*---------------------------------------------------------------------------*
  Name:        Cost
  Description:   Computes the cost of a given data
 *---------------------------------------------------------------------------*/
	double NeuralNet::Cost(const DatasetCreator::Dataset& data) {
		double cost = 0.;

		//iterate through the dataset
		for (unsigned i = 0u; i < data.first.size(); i++) {
			const std::vector<double> computedOutput = ForwardPropagation(data.first[i]);
			const std::vector<double>& datasetOutput = data.second[i];

			for (unsigned k = 0u; k < datasetOutput.size(); k++) {
				cost += (computedOutput[k] - datasetOutput[k]) * (computedOutput[k] - datasetOutput[k]);
			}

		}
		return cost / (2. * data.first.size());
	}

/*---------------------------------------------------------------------------*
  Name:        ComputeValueByActivationFunction
  Description:   Computes value based on activation function
 *---------------------------------------------------------------------------*/
	double NeuralNet::ComputeValueByActivationFunction(const double& value, ActivationFunction::Type functionType) {
		switch (functionType) {
		case ActivationFunction::Type::eNone:
			return ActivationFunction::None(value);
		case ActivationFunction::Type::eSigmoid:
			return ActivationFunction::Sigmoid(value);
		case ActivationFunction::Type::eTanh:
			return ActivationFunction::Tanh(value);
		case ActivationFunction::Type::eRelu:
			return ActivationFunction::Relu(value);
		}
	}

	/*---------------------------------------------------------------------------*
	  Name:        ComputeDerivativeValueByActivationFunction
	  Description:   Computes derivative value based on activation function
	 *---------------------------------------------------------------------------*/
	double NeuralNet::ComputeDerivativeValueByActivationFunction(const double& value, ActivationFunction::Type functionType) {
		switch (functionType) {
		case ActivationFunction::Type::eNone:
			return value;
		case ActivationFunction::Type::eSigmoid:
			return (1.0 - value) * value;
		case ActivationFunction::Type::eTanh:
			return 1.0 - value * value;
		case ActivationFunction::Type::eRelu:
			return value >= 0. ? 1. : 0.;
		}
	}
}