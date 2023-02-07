//---------------------------------------------------------------------------
#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H
//---------------------------------------------------------------------------


namespace CS397
{
class IsolationForest
{
  public:
    // constructor will create the isolation forest for the given data
    IsolationForest(const Dataset & data, // dataset to create the forest for
                    unsigned        numTrees);   // number of tree to generate in the forest

    // destructor will free all the memory of the trees
    ~IsolationForest();

    // returns  the anomaly score of the given input for the create isolation forest
    double ComputeScore(const std::vector<double> & input) const;

    // remove copy and assignment operator
    IsolationForest(const IsolationForest &) = delete;
    void operator=(const IsolationForest &) = delete;

  private:
    // node of isolation tree
    struct iTreeNode
    {
        // isolation tree node constructor (selects random split axis and value)
        iTreeNode(const Dataset & d);

        iTreeNode * left;
        iTreeNode * right;
        unsigned    splitAxis;
        double      splitValue;
        Dataset     data;
    };

    // recursive tree generation
    static iTreeNode * GenerateTree(const Dataset & data);
    // clean up of all the nodes in the tree
    static void FreeTree(iTreeNode * node);
    // computes the path length for the given datapoint (input) on the given node recursively
    static unsigned PathLength(const iTreeNode * node, const std::vector<double> & input);

    // dataset to create the forest for
    std::vector<std::vector<double>> mDataset;
    std::vector<iTreeNode *>         mTreeRoots;
};
} // namespace CS397

#endif