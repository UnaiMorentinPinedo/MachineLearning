/*-------------------------------------------------------
File Name: MarcovDecisionProcess.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "MarkovDecisionProcess.h"


namespace CS397 {
/*---------------------------------------------------------------------------*
  Name:         MarkovDecisionProcess
  Description:  Constructor receives all the data to execute the algorithm
 *---------------------------------------------------------------------------*/
	MarkovDecisionProcess::MarkovDecisionProcess(const std::vector<MarkovState>& states, const std::vector<MarkovAction>& actions,
		double discountFactor) : mStates(states), mActions(actions), mDiscountFactor(discountFactor)
	{
		//intialize the states value to the reward value
		for (unsigned i = 0u; i < states.size(); i++) {
			mComputedStatesValue.push_back(0.0);
			mComputedActionsValue.push_back(0u);
		}
	}

/*---------------------------------------------------------------------------*
  Name:         Iteration
  Description: Iterates on the state values and best policy once
 *---------------------------------------------------------------------------*/
	bool MarkovDecisionProcess::Iteration() {
		//create temporary vectors
		std::vector<double> temp_state_values(mStates.size());
		std::vector<unsigned> temp_action_values(mStates.size());

		//Iterate through all states
		for (unsigned s = 0u; s < mStates.size(); s++) {
			double maxValue = -10000.0;
			unsigned bestAction_idx = 0u;
			//Iterate through all actions
			for (unsigned a = 0u; a < mActions.size(); a++) {
				const TransitionMatrix& actionTransMtx = mActions[a].mTransitionMat;
				double value = 0.0;

				// compute the actual value for this state and this action
				for (unsigned next_s = 0u; next_s < actionTransMtx.mSize; next_s++) {
					auto debug1 = actionTransMtx.mValues[s * actionTransMtx.mSize + next_s];
					auto debug2 = mComputedStatesValue[next_s];
					auto debug3 = actionTransMtx.mValues[s * actionTransMtx.mSize + next_s] * mComputedStatesValue[next_s];
					value += actionTransMtx.mValues[s * actionTransMtx.mSize + next_s] * mComputedStatesValue[next_s];
				}

				//multiply by discount factor
				value = mStates[s].mReward + mDiscountFactor * value;

				//check if this is the best value achieved until now
				if (value > maxValue) {
					maxValue = value;
					bestAction_idx = a;
				}
			}

			//compute and store this state best action value
			temp_state_values[s] =  maxValue;
			temp_action_values[s] = bestAction_idx;
		}

		//check if the policy has not changed
		bool policyChanged = false;
		for (unsigned i = 0; i < temp_action_values.size(); i++) {
			if (temp_action_values[i] != mComputedActionsValue[i]) {
				policyChanged = true;
				break;
			}
		}

		//if policy has not changed: check if values are barely changing, if so stop iterating
		double diff = 0.0;
		if (policyChanged == false) {
			for (unsigned i = 0u; i < temp_state_values.size(); i++) {
				if (mStates[i].mStatic)
					continue;

				diff += std::abs(temp_state_values[i] - mComputedStatesValue[i]);
			}
		}

		//compy the temporary data
		for (unsigned i = 0u; i < temp_state_values.size(); i++) {
			//state is static
			if (mStates[i].mStatic) 
				mComputedStatesValue[i] = mStates[i].mReward;
			else
				mComputedStatesValue[i] = temp_state_values[i];
		}
		mComputedActionsValue = temp_action_values;

		return policyChanged || diff > c_epsilon;
	}

/*---------------------------------------------------------------------------*
  Name:         GetStateValues
  Description: Returns the last computed values of each state
 *---------------------------------------------------------------------------*/
	std::vector<double> MarkovDecisionProcess::GetStateValues() const {
		return mComputedStatesValue;
	}

/*---------------------------------------------------------------------------*
  Name:         GetBestPolicy
  Description: Returns the last computed best policy (action to take in each state)
 *---------------------------------------------------------------------------*/
	std::vector<unsigned> MarkovDecisionProcess::GetBestPolicy() const {
		return mComputedActionsValue;
	}
}