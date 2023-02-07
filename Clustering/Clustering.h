//---------------------------------------------------------------------------
#ifndef CLUSTERING_H
#define CLUSTERING_H
//---------------------------------------------------------------------------

namespace CS397
{
    struct NormalizationData {
        double max, min, mean;
    };


    class KMeans
    {
    public:
        // Constructor
        KMeans(const Dataset& data,             // Dataset clusters will generated for
            const std::vector<std::vector<double>>& initialCentroids, // Initial centroid positions (number of clusters to generate is extracted from size)
            bool                                     meanNormalization);  // True if input data should be mean normalized

     // Given a single or multiple input datapoint(s), computes the index/indices
     // of the cluster each of the inputs belongs to
        std::vector<unsigned> Predict(const Dataset& input) const;
        unsigned              Predict(const std::vector<double>& input) const;

        // clusters will be adjusted for the dataset
        bool Iteration(double minDisplacement = 0.01);

        // Computes the cost of an external dataset
        double Cost(const Dataset& input);

    private:
        Dataset mDataset;
        std::vector<std::vector<double>> mCentroids;
        std::vector<unsigned> prevCentroidIndexes;
        std::vector<unsigned> numberOfSamplesOnEachCentroidLastFrame;
        bool mMeanNormalization;
        unsigned mNumberOfCentroids;
        bool mTrained;
        std::vector<NormalizationData> mNormalizationData;


        double ComputeDistanceToCentroid(const std::vector<double>& x, const std::vector<double>& centroid) const;
        double ComputeDistanceToCluster(const std::vector<double>& x, const unsigned centroidIndex) const;
    };

    class FuzzyCMeans
    {
    private:
        static std::vector<double> InitialProbabilityMatrix(size_t r, size_t c)
        {
            std::vector<double> mat(r * c);

            // initialize probability matrix to random probabilities
            for (size_t i = 0; i < r; i++)
            {
                double total = 0.0;

                // give random probabilities to each cluster
                for (size_t j = 0; j < c; j++)
                {
                    double value = static_cast<double>(std::rand()) / RAND_MAX;

                    mat[i * c + j] = value;

                    total += value;
                }

                // since all probabilities need to add up to 1 normalize probabilities
                for (unsigned j = 0; j < c; j++)
                {
                    mat[i * c + j] /= total;
                }
            }

            return mat;
        }

    public:
        FuzzyCMeans(const Dataset& data,             // Dataset clusters will generated for
            const std::vector<std::vector<double>>& initialCentroids, // Initial centroid positions (number of clusters to generate is extracted from size)
            double                                   fuzziness,        // Specifies the fuzziness applied when learning
            bool                                     meanNormalization); // True if input data should be mean normalized

// Given a single or multiple input datapoint(s), computes the probabilities of 
// belonging to a cluster for each of the inputs
        std::vector<std::vector<double>> Predict(const Dataset& input) const;
        std::vector<double>              Predict(const std::vector<double>& input) const;

        // clusters will be adjusted for the dataset
        bool Iteration(double minDisplacement = 0.01);

        // Computes the cost of an external dataset
        double Cost(const Dataset& input);


    private:
        std::vector<double> mMatrix;
        Dataset mDataset;
        std::vector<std::vector<double>> mCentroids;
        bool mMeanNormalization;
        double mFuzziness;
        unsigned mNumberOfCentroids;
        std::vector<NormalizationData> mNormalizationData;

        void UpdateCentroidsPosition();
        double ComputeDistance(const std::vector<double>& x, const std::vector<double>& centroid) const;
        double ComputeDistanceSquared(const std::vector<double>& x, const std::vector<double>& centroid) const;
    };
} // namespace CS397

#endif