/*-------------------------------------------------------
File Name: MarkovChain.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "MarkovChain.h"
#include <cmath>

namespace CS397 {
/*---------------------------------------------------------------------------*
  Name:         MarkovChain
  Description:  Constructor receives all the data to execute the algorithm
 *---------------------------------------------------------------------------*/
	MarkovChain::MarkovChain(const std::vector<MarkovState>& states, const std::vector<double>& transitionMat,
		double discountFactor) 
		: mStates(states), mTransitionMtx(transitionMat), mDiscountFactor(discountFactor), mNumberofStates(states.size())
	{
		//intialize the states value to the reward value
		for (unsigned i = 0u; i < states.size(); i++)
			mComputedStatesValue.push_back(0.0);
	}

/*---------------------------------------------------------------------------*
  Name:         Iteration
  Description:  iterates on the state values once
 *---------------------------------------------------------------------------*/
	bool MarkovChain::Iteration() {
		std::vector<double> temp_state_values;

		//do the Markov Chain computation
		for (unsigned i = 0u; i < mStates.size(); i++) {
			double value = 0.0;
			for (unsigned k = 0u; k < mStates.size(); k++) {
				value += mTransitionMtx[i * static_cast<int>(mStates.size()) + k] * mComputedStatesValue[k];
			}

			temp_state_values.push_back(mStates[i].mReward + mDiscountFactor * value);
		}

		//check if values are barely changing, if so stop iterating
		double diff = 0.0;
		for (unsigned i = 0u; i < temp_state_values.size(); i++)
			diff += std::abs(temp_state_values[i] - mComputedStatesValue[i]);

		//store in member vector the temporal data
		mComputedStatesValue = temp_state_values;

		//check computed diff with a very small number in order to continue iterating
		return diff > c_epsilon;
	}

/*---------------------------------------------------------------------------*
  Name:         GetProbabilityNTransitions
  Description:  Computes the probability of being in each state after the amount of transitions specified in the parameter (numTransitions)
				 and given an initial probability to each state
 *---------------------------------------------------------------------------*/
	std::vector<double> MarkovChain::GetProbabilityNTransitions(const std::vector<double> initialProbabilities, unsigned numTransitions) const {
		std::vector<double> probMatrix = mTransitionMtx;

		//do matrix multiplication
		for (unsigned i = 0u; i < numTransitions; i++) {
			//update probability matrix
			probMatrix = Matrix_Matrix_Multiplication(probMatrix, mTransitionMtx);
		}

		//perform vector multiplication
		std::vector<double> finalProb = Vector_Matrix_Multiplication(initialProbabilities, probMatrix);

		return finalProb;
	}

/*---------------------------------------------------------------------------*
  Name:         GetStateValues
  Description:   returns the last computed values of each state
 *---------------------------------------------------------------------------*/
	std::vector<double> MarkovChain::GetStateValues() const {
		return mComputedStatesValue;
	}

/*---------------------------------------------------------------------------*
  Name:         Matrix_Matrix_Multiplication
  Description:  Squared matrix multiplication
 *---------------------------------------------------------------------------*/
	std::vector<double> MarkovChain::Matrix_Matrix_Multiplication(const std::vector<double>& mtx_0, const std::vector<double>& mtx_1) const{
		//create resulting matrix
		std::vector<double> result(mNumberofStates * mNumberofStates);

		for (unsigned r = 0u; r < mNumberofStates; r++) {

			for (unsigned m_r = 0u; m_r < mNumberofStates; m_r++) {
				double value = 0.0;

				for (unsigned m_c = 0u; m_c < mNumberofStates; m_c++)
					value += mtx_0[r * mNumberofStates + m_c] * mtx_1[m_c * mNumberofStates + m_r];

				result[r * mNumberofStates + m_r] = value;
			}
		}

		return result;
	}

/*---------------------------------------------------------------------------*
  Name:         Vector_Matrix_Multiplication
  Description:  vector * matrix
 *---------------------------------------------------------------------------*/
	std::vector<double> MarkovChain::Vector_Matrix_Multiplication(const std::vector<double>& vec, const std::vector<double>& mtx) const {
		//create resulting matrix
		std::vector<double> result;

		//multiply initiaProbabily matrix by the computed transition matrix
		for (unsigned r = 0u; r < mNumberofStates; r++) {
			double value = 0.0;
			for (unsigned c = 0u; c < mNumberofStates; c++) {
				value += vec[c] * mtx[r + mNumberofStates * c];
			}

			result.push_back(value);
		}

		return result;
	}
}