/*-------------------------------------------------------
File Name: Regression.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "Regression.h"

namespace CS397 {
	/*---------------------------------------------------------------------------*
	  Name:         Regression
	  Description:  Constructor of the class Regression
	 *---------------------------------------------------------------------------*/
	Regression::Regression(const Dataset& dataset, const std::vector<Feature>& features, double lr, bool meanNormalization) :
	mDataset(dataset), mFeatures(features), mLearningRate(lr), mMeanNormalization(meanNormalization){
		if (mMeanNormalization) {
			for (unsigned i = 0u; i < mFeatures.size(); i++) {
				if (mFeatures[i].inputIdx < 0)
					continue;

				mFeatures[i].mean = 0.0;
				mFeatures[i].max = -1000000.0;
				mFeatures[i].min = 1000000.0;

				//add all input values and check for min/max values
				for (unsigned k = 0u; k < mDataset.input.size(); k++) {
					double& input_value = mDataset.input[k][mFeatures[i].inputIdx];
					mFeatures[i].mean += input_value;

					if (input_value < mFeatures[i].min)
						mFeatures[i].min = input_value;
					if (input_value > mFeatures[i].max)
						mFeatures[i].max = input_value;
				}

				//divide it by number of inputs
				mFeatures[i].mean /= static_cast<int>(mDataset.input.size());
			}
		}
	}

	/*---------------------------------------------------------------------------*
	  Name:         Predict
	  Description:  Predict the output of given data as parameter
	 *---------------------------------------------------------------------------*/
	double Regression::Predict(const std::vector<double>& input) const {
		double value = 0.;

		for (unsigned k = 0u; k < mFeatures.size(); k++) {
			const Feature& feature = mFeatures[k];
			for (unsigned i = 0u; i < input.size(); i++) {
				//get the feature for easier acces
				if (feature.inputIdx < 0)
					value += feature.theta;
				else {
					//mean normalize if flaged
					if (mMeanNormalization) {
						double normValue = (input[feature.inputIdx] - feature.mean) / (feature.max - feature.min);
						value += feature.theta * std::pow(normValue, feature.power);
					}
					else
						value += feature.theta * std::pow(input[feature.inputIdx], feature.power);
				}
			}
		}

		return value;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Predict
	  Description:  Predict the output of given data as parameter calling above function
	 *---------------------------------------------------------------------------*/
	std::vector<double> Regression::Predict(const std::vector<std::vector<double>>& input) const {
		std::vector<double> output;

		for (unsigned i = 0u; i < input.size(); i++)
			output.push_back(Predict(input[i]));

		return output;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Cost
	  Description:  Computes the cost of given output and target parameter
	 *---------------------------------------------------------------------------*/
	double Regression::Cost(const std::vector<double>& output, const std::vector<double>& target) const {
		double costValue = 0.;

		for (unsigned i = 0; i < output.size(); i++)
			costValue += (output[i] - target[i]) * (output[i] - target[i]);

		return costValue / (2.0 * static_cast<int>(output.size()));
	}

	/*---------------------------------------------------------------------------*
	  Name:         Iteration
	  Description:  Main logic of the algorithm, called many times
	 *---------------------------------------------------------------------------*/
	bool Regression::Iteration(double minDerivative) {
		//predict
		std::vector<double> predicted_output = Predict(mDataset.input);

		std::vector<double> tempThetaValues;
		bool reachedMinimum = true;

		//compute cost derivative for each feature
		for (unsigned i = 0u; i < mFeatures.size(); i++) {
			double costDerivate = CostDerivate(predicted_output, i);

			// we are done, we found a minimum
			if (std::abs(costDerivate) < minDerivative) {
				tempThetaValues.push_back(mFeatures[i].theta);
				continue;
			}

			reachedMinimum = false;

			//adjust the theta value fo the features depending of the value of the cost derivative
			tempThetaValues.push_back(mFeatures[i].theta - mLearningRate * costDerivate);
		}

		// we are done iterating
		if (reachedMinimum)
			return false;

		//set new values of theta
		for (unsigned i = 0u; i < tempThetaValues.size(); i++)
			mFeatures[i].theta = tempThetaValues[i];

		return true;
	}

	/*---------------------------------------------------------------------------*
	  Name:         CostDerivate
	  Description:  Computes the cost derivativee
	 *---------------------------------------------------------------------------*/
	double Regression::CostDerivate(const std::vector<double>& predicted_output, unsigned feature_idx) const {
		double value = 0.;

		for (unsigned i = 0u; i < mDataset.input.size(); i++) {
			double sub = predicted_output[i] - mDataset.output[i];

			//mulitply by derivative chain rule
			if (mFeatures[feature_idx].inputIdx >= 0) {
				if (mMeanNormalization) {
					double normValue = (mDataset.input[i][mFeatures[feature_idx].inputIdx] - mFeatures[feature_idx].mean) / 
						(mFeatures[feature_idx].max - mFeatures[feature_idx].min);
					value += sub * std::pow(normValue,
						mFeatures[feature_idx].power);
				}
				else {
					value += sub * std::pow(mDataset.input[i][mFeatures[feature_idx].inputIdx],
						mFeatures[feature_idx].power);
				}
			}
			else
				value += sub;
		}

		return value / static_cast<int>(mDataset.input.size());
	}
}