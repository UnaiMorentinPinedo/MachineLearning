//---------------------------------------------------------------------------
#ifndef MARKOV_UTILS_H
#define MARKOV_UTILS_H
//---------------------------------------------------------------------------

#include <string>
#include <vector>

namespace CS397
{

static double c_epsilon = 0.0001;

//  transition matrix contains probabilities of transitionion from each state to any other state
struct TransitionMatrix
{
  public:
    unsigned            mSize;
    std::vector<double> mValues;
};

//  markov state structure that stores the reward of reaching that state
struct MarkovState
{
    std::string mName;
    double      mReward;
    bool        mStatic = false;
};

// markov action structure that stores the probabilities to transition to each state given an action
struct MarkovAction
{
    std::string      mName;
    TransitionMatrix mTransitionMat;
};

} // namespace CS397

#endif