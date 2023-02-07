//---------------------------------------------------------------------------
#ifndef MARKOV_DECISION_PROCESS_H
#define MARKOV_DECISION_PROCESS_H
//---------------------------------------------------------------------------

#include "MarkovUtils.h"

namespace CS397
{

class MarkovDecisionProcess
{
  public:
    // constructor receives all the data to execute the algorithm
    MarkovDecisionProcess(const std::vector<MarkovState> &  states,  // container of the different states to analyze
                          const std::vector<MarkovAction> & actions, // actions that can be taken (they store their respective transition matrices)
                          double                            discountFactor);                    // discount factor that reduces the rewards every transition

    // iterates on the state values and best policy once
    bool Iteration();

    // returns the last computed values of each state
    std::vector<double> GetStateValues() const;
    // returns the last computed best policy (action to take in each state)
    std::vector<unsigned> GetBestPolicy() const;

private:
    std::vector<MarkovState>        mStates;
    std::vector<MarkovAction>       mActions;
    std::vector<double>             mComputedStatesValue;
    std::vector<unsigned>           mComputedActionsValue;
    double                          mDiscountFactor;    
};

} // namespace CS397

#endif