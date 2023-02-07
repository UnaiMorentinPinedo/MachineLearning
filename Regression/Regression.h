//---------------------------------------------------------------------------
#ifndef REGRESSION_H
#define REGRESSION_H
//---------------------------------------------------------------------------

namespace CS397
{

struct Feature
{
    int    inputIdx; // Index of the input this feature refers to (negative index means intercept)
    int    power;    // Power of the feature
    double theta;    // Constant multiplier of the feature (what the algorithm will adjust)
    double max, min, mean;
};

class Regression
{
  public:
    // Constructor
    Regression(const Dataset &              dataset,  // Dataset that the regression will learn from
               const std::vector<Feature> & features, // Features
               double                       lr,       // Learning rate fo the linear regression
               bool                         meanNormalization);               // True if input data should be mean normalized

    // Given a single or multiple input datapoint(s), computes the value(s) that the linear regression would predict
    double              Predict(const std::vector<double> & input) const;
    std::vector<double> Predict(const std::vector<std::vector<double>> & input) const;

    // Given the output predicted and the actual real value (target), computes the cost
    double Cost(const std::vector<double> & output,
                const std::vector<double> & target) const;

    // Linear regression values will be adjusted for the dataset using gradient descent.
    // Returns false when all derivatives are smaller than the minimum provided.
    bool Iteration(double minDerivative = 0.01);

  private:
      double CostDerivate(const std::vector<double>& output, unsigned feature_idx) const;

      Dataset mDataset;
      std::vector<Feature> mFeatures;
      double mLearningRate;
      bool mMeanNormalization;
};

} // namespace CS397
#endif