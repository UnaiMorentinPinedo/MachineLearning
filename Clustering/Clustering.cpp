/*-------------------------------------------------------
File Name: Clustering.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "Clustering.h"
#include <cmath>


namespace CS397 {
/*---------------------------------------------------------------------------*
  Name:         KMeans
  Description:  Constructor of the class Kmeans
 *---------------------------------------------------------------------------*/
	KMeans::KMeans(const Dataset& data, const std::vector<std::vector<double>>& initialCentroids,
		bool meanNormalization) : 
		mDataset(data), 
		mCentroids(initialCentroids),
		mMeanNormalization(meanNormalization),
		mTrained(false),
		mNumberOfCentroids(initialCentroids.size()){

		if (mMeanNormalization) {
			mNormalizationData.resize(data[0].size());
			for (unsigned i = 0u; i < mNormalizationData.size(); i++) {
				mNormalizationData[i].mean = 0.0;
				mNormalizationData[i].max = -1000000.0;
				mNormalizationData[i].min = 1000000.0;

				//add all input values and check for min/max values
				for (unsigned k = 0u; k < mDataset.size(); k++) {
					double& input_value = mDataset[k][i];
					mNormalizationData[i].mean += input_value;

					if (input_value < mNormalizationData[i].min)
						mNormalizationData[i].min = input_value;
					if (input_value > mNormalizationData[i].max)
						mNormalizationData[i].max = input_value;
				}

				//divide it by number of inputs
				mNormalizationData[i].mean /= static_cast<int>(mDataset.size());
			}
			//normalize centroid positions
			for (unsigned c = 0u; c < mCentroids.size(); c++) {
				for (unsigned l = 0u; l < mCentroids[c].size(); l++) {
					mCentroids[c][l] = (mCentroids[c][l] - mNormalizationData[l].mean) / (mNormalizationData[l].max - mNormalizationData[l].min);
				}
			}
		}
	}

/*---------------------------------------------------------------------------*
  Name:         Predict
  Description:  Predicts the indexes of given input
 *---------------------------------------------------------------------------*/
	std::vector<unsigned> KMeans::Predict(const Dataset& input) const {
		std::vector<unsigned> centroidIndex;

		for (unsigned i = 0u; i < input.size(); i++) {
			centroidIndex.push_back(Predict(input[i]));
		}

		return centroidIndex;
	}

/*---------------------------------------------------------------------------*
  Name:         Predict
  Description:  Predicts the indexes of given input
 *---------------------------------------------------------------------------*/
	unsigned KMeans::Predict(const std::vector<double>& input) const {
		double minDist = 10000000000.0; //very big number
		unsigned minDistIndex = -1;

		for (unsigned i = 0u; i < mCentroids.size(); i++) {
			double dist;
			if(mTrained)
				dist = ComputeDistanceToCluster(input, i);
			else
				dist = ComputeDistanceToCentroid(input, mCentroids[i]);

			//compare the current distance with the currently minimum one
			if (dist < minDist) {
				minDist = dist;
				minDistIndex = i;
			}
		}

		return minDistIndex;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Iteration
	  Description:  Main logic of the algorithm
	 *---------------------------------------------------------------------------*/
	bool KMeans::Iteration(double minDisplacement) {
		//create vectors to compute new centroid positions
		std::vector<std::vector<double>> centroidsNewPos(mNumberOfCentroids);
		std::vector<unsigned> numberOfSamplesOnEachCentroid;

		//initialize those vectors
		for (unsigned i = 0u; i < mNumberOfCentroids; i++) {
			for (unsigned k = 0u; k < mDataset[0].size(); k++) {
				centroidsNewPos[i].push_back(0.0);
			}
			numberOfSamplesOnEachCentroid.push_back(0u);
		}

		//call predict to know the input to which centroid is assigned
		std::vector<unsigned> centroidIndexes = Predict(mDataset);

		//compute centroid new position
		for (unsigned i = 0u; i < mDataset.size(); i++) {
			//get the index of the centroid ofr the data of this loop iteration
			unsigned iterationCentroidIndex = centroidIndexes[i];

			numberOfSamplesOnEachCentroid[iterationCentroidIndex]++; //add one to the count of nuumber of samples of the centroid

			for (unsigned k = 0u; k < centroidsNewPos[iterationCentroidIndex].size(); k++) {
				if (mMeanNormalization)
					centroidsNewPos[iterationCentroidIndex][k] += (mDataset[i][k]  - mNormalizationData[k].mean)
					/ (mNormalizationData[k].max - mNormalizationData[k].min);
				else
					centroidsNewPos[iterationCentroidIndex][k] += mDataset[i][k];
			}
		}

		//average the new position of the centroid
		for (unsigned i = 0u; i < mNumberOfCentroids; i++) {
			for(unsigned k = 0u; k < mCentroids[i].size(); k++)
				mCentroids[i][k] = centroidsNewPos[i][k] / numberOfSamplesOnEachCentroid[i];
		}

		//check if we are done, only when no sample changes from cluster in one iteration
		if (numberOfSamplesOnEachCentroidLastFrame.empty()) {
			prevCentroidIndexes = centroidIndexes;
			numberOfSamplesOnEachCentroidLastFrame = numberOfSamplesOnEachCentroid;
			return true;
		}
		else {
			//first quicker way to know the index of each sample is different this frame
			bool sameNumberOfSamplesInEachCluster = true;
			for (unsigned i = 0u; i < mNumberOfCentroids; i++) {
				if (numberOfSamplesOnEachCentroid[i] != numberOfSamplesOnEachCentroidLastFrame[i])
					sameNumberOfSamplesInEachCluster = false;
			}

			// if same number of samples do a deeper research and check each sample previous and new cluster index
			if (sameNumberOfSamplesInEachCluster) {
				for (unsigned i = 0u; i < centroidIndexes.size(); i++) {
					if (prevCentroidIndexes[i] != centroidIndexes[i])
						return true;
				}
			}

			//update vectors
			prevCentroidIndexes = centroidIndexes;
			numberOfSamplesOnEachCentroidLastFrame = numberOfSamplesOnEachCentroid;

			mTrained = true; //we are done with the traingin step
			return false;
		}
	}

/*---------------------------------------------------------------------------*
  Name:         Cost
  Description:  Computes the cost of the given input
 *---------------------------------------------------------------------------*/
	double KMeans::Cost(const Dataset& input) {
		mTrained = false;
		//call predict to know the input to which centroid is assigned
		std::vector<unsigned> centroidIndex = Predict(input);

		double cost = 0.0;

		for (unsigned i = 0u; i < input.size(); i++) {
			cost += ComputeDistanceToCentroid(input[i], mCentroids[centroidIndex[i]]);
		}

		return cost/static_cast<int>(input.size());
	}

/*---------------------------------------------------------------------------*
  Name:         ComputeDistanceToCentroid
  Description:  Computes the distance from input to centroid
 *---------------------------------------------------------------------------*/
	double KMeans::ComputeDistanceToCentroid(const std::vector<double>& x, const std::vector<double>& centroid) const {
		double dist = 0.0;
		for (unsigned i = 0u; i < x.size(); i++) {
			if (mMeanNormalization) {
				double normValue = (x[i] - mNormalizationData[i].mean) / (mNormalizationData[i].max - mNormalizationData[i].min);
				dist += (normValue - centroid[i]) * (normValue - centroid[i]);
			}
			else {
				dist += (x[i] - centroid[i]) * (x[i] - centroid[i]);
			}
		}
		return dist;
	}

	/*---------------------------------------------------------------------------*
	  Name:         ComputeDistanceToCluster
	  Description:  Computes the distance from input to any cluster point
	 *---------------------------------------------------------------------------*/
	double KMeans::ComputeDistanceToCluster(const std::vector<double>& x, const unsigned centroidIndex) const {
		double closestDistance = 100000000.0; //very big number to be overwritten

		for (unsigned k = 0u; k < prevCentroidIndexes.size(); k++) {
			if (prevCentroidIndexes[k] != centroidIndex)	continue; //this sample does not belong to the cluster we are cheking with

			//get sample
			const std::vector<double>& data = mDataset[k];

			//compute distance between samples
			double dist = 0.0;
			for (unsigned i = 0u; i < x.size(); i++) {
				if (mMeanNormalization) {
					double normValue_x = (x[i] - mNormalizationData[i].mean) / (mNormalizationData[i].max - mNormalizationData[i].min);
					double normValue_data = (data[i] - mNormalizationData[i].mean) / (mNormalizationData[i].max - mNormalizationData[i].min);
					dist += (normValue_x - normValue_data) * (normValue_x - normValue_data);
				}
				else {
					dist += (x[i] - data[i]) * (x[i] - data[i]);
				}
			}

			//if computed distance is smaller
			if (dist < closestDistance)
				closestDistance = dist;
		}
		return closestDistance;
	}

/*---------------------------------------------------------------------------*
Name:         FuzzyCMeans
Description:  Constructor of the class FuzzyCMeans
*---------------------------------------------------------------------------*/
	FuzzyCMeans::FuzzyCMeans(const Dataset& data, const std::vector<std::vector<double>>& initialCentroids,
		double fuzziness, bool meanNormalization) :
		mDataset(data),
		mCentroids(initialCentroids),
		mMeanNormalization(meanNormalization),
		mFuzziness(fuzziness),
		mNumberOfCentroids(initialCentroids.size()) {

		mMatrix = InitialProbabilityMatrix(data.size(), initialCentroids.size());

		if (mMeanNormalization) {
			mNormalizationData.resize(data[0].size());
			for (unsigned i = 0u; i < mNormalizationData.size(); i++) {
				mNormalizationData[i].mean = 0.0;
				//very big numbers to be overwritten
				mNormalizationData[i].max = -1000000.0;
				mNormalizationData[i].min = 1000000.0;

				//add all input values and check for min/max values
				for (unsigned k = 0; k < mDataset.size(); k++) {
					double& input_value = mDataset[k][i];
					mNormalizationData[i].mean += input_value;

					if (input_value < mNormalizationData[i].min)
						mNormalizationData[i].min = input_value;
					if (input_value > mNormalizationData[i].max)
						mNormalizationData[i].max = input_value;
				}

				//divide it by number of inputs
				mNormalizationData[i].mean /= static_cast<int>(mDataset.size());
			}
			//normalize centroid positions
			for (unsigned c = 0u; c < mCentroids.size(); c++) {
				for (unsigned l = 0u; l < mCentroids[c].size(); l++) {
					mCentroids[c][l] = (mCentroids[c][l] - mNormalizationData[l].mean) / (mNormalizationData[l].max - mNormalizationData[l].min);
				}
			}
		}
	}


	/*---------------------------------------------------------------------------*
	  Name:         Predict
	  Description:  Predicts the membership vaklues of given input
	 *---------------------------------------------------------------------------*/
	std::vector<std::vector<double>> FuzzyCMeans::Predict(const Dataset& input) const {
		std::vector<std::vector<double>> membershipMatrix;

		for (unsigned i = 0u; i < input.size(); i++) {
			membershipMatrix.push_back(Predict(input[i]));
		}

		return membershipMatrix;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Predict
	  Description:  Predicts the indexes of given input
	 *---------------------------------------------------------------------------*/
	std::vector<double> FuzzyCMeans::Predict(const std::vector<double>& input) const {
		std::vector<double> membershipMatrix;

		for (unsigned k = 0u; k < mNumberOfCentroids; k++) {
			double sum = 0.0;
			double distToCentroid = ComputeDistance(input, mCentroids[k]);

			//special case
			if (distToCentroid == 0.0) {//x is actually the centroid
				std::vector<double> specialCase;
				for (unsigned s = 0; s < mNumberOfCentroids; s++)
					specialCase.push_back(0.0);
				specialCase[k] = 1.0;
				return specialCase;
			}

			for (unsigned j = 0u; j < mNumberOfCentroids; j++) {
				double distToOtherCentroid = ComputeDistance(input, mCentroids[j]);

				sum += std::pow((distToCentroid / distToOtherCentroid), 2.0/(mFuzziness-1.0));
			}

			membershipMatrix.push_back(1.0/sum);
		}

		return membershipMatrix;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Iteration
	  Description:  Main logic of the algorithm
	 *---------------------------------------------------------------------------*/
	bool FuzzyCMeans::Iteration(double minDisplacement) {
		UpdateCentroidsPosition();

		//update the matrix vector
		std::vector<std::vector<double>> memberShipMatrix = Predict(mDataset);

		double accuracy = 0.;
		for (unsigned i = 0u; i < memberShipMatrix.size(); i++) {
			for (unsigned k = 0u; k < mNumberOfCentroids; k++) {
				double updatedValue = memberShipMatrix[i][k];

				//keep track of accuracy
				double diff = std::abs(updatedValue - mMatrix[i * mNumberOfCentroids + k]);
				if (diff > accuracy)
					accuracy = diff;

				mMatrix[i * mNumberOfCentroids + k] = updatedValue;
			}
		}

		return accuracy > minDisplacement;
	}

	/*---------------------------------------------------------------------------*
	  Name:         Cost
	  Description:  Computes the cost of the given input
	 *---------------------------------------------------------------------------*/
	double FuzzyCMeans::Cost(const Dataset& input) {
		std::vector<std::vector<double>> membershipMatrix = Predict(input);

		double cost = 0.0;

		for (unsigned i = 0u; i < input.size(); i++) {
			double iterationInputCost = 0.0;
			for (unsigned c = 0; c < mNumberOfCentroids; c++) {
				double membershipValue = std::pow(membershipMatrix[i][c], mFuzziness);
				double distToCentroid = ComputeDistanceSquared(input[i], mCentroids[c]);

				iterationInputCost += membershipValue * distToCentroid;
			}

			cost += iterationInputCost;
		}

		return cost / static_cast<int>(input.size());
	}

	void FuzzyCMeans::UpdateCentroidsPosition() {
		for (unsigned c = 0u; c < mNumberOfCentroids; c++) {
			for (unsigned k = 0u; k < mCentroids[c].size(); k++) {
				double topValue = 0.0;
				double botValue = 0.0;

				for (unsigned i = 0u; i < mDataset.size(); i++) {
					double value = mDataset[i][k];
					if(mMeanNormalization)
						value = (value - mNormalizationData[k].mean) / (mNormalizationData[k].max - mNormalizationData[k].min);

					double membershipValue = std::pow(mMatrix[mNumberOfCentroids * i + c], mFuzziness);
					topValue += membershipValue * value;
					botValue += membershipValue;
				}

				mCentroids[c][k] = topValue / botValue;
			}
		}
	}

/*---------------------------------------------------------------------------*
 Name:         ComputeDistance
 Description:  Computes the distance from input to centroid
*---------------------------------------------------------------------------*/
	double FuzzyCMeans::ComputeDistance(const std::vector<double>& x, const std::vector<double>& centroid) const {
		double dist = 0.0;
		for (unsigned i = 0; i < x.size(); i++) {
			double value = x[i];
			if(mMeanNormalization)
				value = (value - mNormalizationData[i].mean) / (mNormalizationData[i].max - mNormalizationData[i].min);

			dist += (value - centroid[i]) * (value - centroid[i]);
		}

		return sqrt(dist);
	}

/*---------------------------------------------------------------------------*
  Name:         ComputeDistanceSquared
  Description:  Computes the distance sqaured from input to centroid
 *---------------------------------------------------------------------------*/
	double FuzzyCMeans::ComputeDistanceSquared(const std::vector<double>& x, const std::vector<double>& centroid) const {
		double dist = 0.0;
		for (unsigned i = 0; i < x.size(); i++) {
			double value = x[i];
			if (mMeanNormalization)
				value = (value - mNormalizationData[i].mean) / (mNormalizationData[i].max - mNormalizationData[i].min);

			dist += (value - centroid[i]) * (value - centroid[i]);
		}

		return dist;
	}
}