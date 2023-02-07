//---------------------------------------------------------------------------
#ifndef DECISION_TREE_H
#define DECISION_TREE_H
//---------------------------------------------------------------------------

#include <string>
#include <vector>

namespace CS397
{

using Dataset = std::vector<std::pair<std::vector<std::string>, std::string>>;

class DecisionTree
{
  public:
    // constructor will create the decision tree with specified max depth for the given data
    // negative max depth means expanding the full tree
    DecisionTree(const Dataset & data, const std::vector<std::string> & attributeNames, int maxDepth);

    // destructor will free all the memory of the trees
    ~DecisionTree();

    // remove copy and assignment operator
    DecisionTree(const DecisionTree &) = delete;
    void operator=(const DecisionTree &) = delete;

    std::string ToString() const;

  private:
    // node of isolation tree
    struct DecisionTreeNode
    {
        // isolation tree node constructor (selects random split axis and value)
        DecisionTreeNode(const Dataset & d, const DecisionTreeNode * p);

        // children nodes
        std::vector<DecisionTreeNode *> children;
        const DecisionTreeNode *        parent;

        Dataset data;

        // stores relevant value for each node
        union
        {
            int outputClassIndex; // leaf
            int attributeIndex;   // internal
        };
    };
    
    // helper recursive function for the string formatting
    std::string RecursiveToString(const DecisionTreeNode * node, unsigned tabs) const;

    // Reads all the attributes and the possible values from dataset
    // and stores them in the member variable mAttributes
    void ExtractAttributes(const std::vector<std::string> & attributeNames);
    
    // recursive tree generation
    DecisionTreeNode * GenerateTree(const Dataset & data, const DecisionTreeNode * parent, int depth);
    // clean up of all the nodes in the tree
    void FreeTree(DecisionTreeNode * node);

    // selects and attribute based on the information gain (ID3) of the data
    int                  SelectAttribute(const Dataset & data);  
    // selects the most probable output class on the given node (leaf node) 
    int                  SelectOutputClass(const DecisionTreeNode * node);
    // split given dataset based on the selected attribute (one dataset per value of the attribute)
    std::vector<Dataset> SplitDataset(const Dataset & data, int attributeIndex) const;  

    // dataset to create the forest for
    Dataset            mDataset;
    DecisionTreeNode * mRoot;

    struct Attribute
    {
        std::string              name;
        std::vector<std::string> values;
    };
    std::vector<Attribute>   mAttributes;
    std::vector<std::string> mOutputClasses;
    int mMaxDepth;

    double ComputeEntropy(const unsigned yes_values, const unsigned total_values);
    bool CheckEndConditions(const Dataset& data, std::vector<std::pair<Attribute, bool>>& possibleAttributes);
    unsigned ChooseOutputValue(const Dataset& data);
};

} // namespace CS397

#endif