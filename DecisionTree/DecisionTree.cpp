/*-------------------------------------------------------
File Name: DecisionTree.cpp
Author: Unai Morentin
---------------------------------------------------------*/


#include "DecisionTree.h"

#include <algorithm>
#include <cassert>

namespace CS397
{

// constructor will create the decision tree for the given data
DecisionTree::DecisionTree(const Dataset & data, const std::vector<std::string> & attributeNames, int maxDepth) :
    mDataset(data),
    mRoot(nullptr),
    mMaxDepth(maxDepth)
{
    ExtractAttributes(attributeNames);
    // generate the forest by creating the trees
    mRoot = GenerateTree(data, nullptr, maxDepth);
}

// destructor will free all the memory of the trees
DecisionTree::~DecisionTree()
{
    // free every tree in the forest
    FreeTree(mRoot);
}

// converts the generated tree into string format
std::string DecisionTree::ToString() const
{
    return RecursiveToString(mRoot, 0);
}

// helper recursive function for the string formatting
std::string DecisionTree::RecursiveToString(const DecisionTreeNode * node, unsigned tabs) const
{
    std::string output;

    // add formatting tabs
    for (unsigned i = 0; i < tabs; i++)
    {
        output += "   ";
    }

    // leaf node only writes output class
    if (node->children.size() == 0)
    {
        if (node->outputClassIndex >= 0)
        {
            output += "OutputClass: " + mOutputClasses[node->outputClassIndex] + "\n";        
        }
    }
    // output splitting attribute and recursively format children
    else
    {
        output += "ATTRIBUTE: " + mAttributes[node->attributeIndex].name + "\n";

        for (unsigned i = 0; i < node->children.size(); i++)
        {
            for (unsigned i = 0; i < tabs; i++)
            {
                output += "   ";
            }
            output += " Value: " + mAttributes[node->attributeIndex].values[i] + "\n";
            output += RecursiveToString(node->children[i], tabs + 1);
        }
    }

    return output;
}

// Reads all the attributes and the possible values from dataset
// and stores them in the member variable mAttributes
void DecisionTree::ExtractAttributes(const std::vector<std::string> & attributeNames)
{
    assert(mDataset.size() > 0);

    mAttributes.resize(mDataset[0].first.size());

    // traverse the dataset to get all possible values for each attribute
    for (unsigned i = 0; i < mDataset.size(); i++)
    {
        for (unsigned a = 0; a < mAttributes.size(); a++)
        {
            mAttributes[a].name = attributeNames[a];

            const std::string & value  = mDataset[i].first[a];
            const auto          itFind = std::find(mAttributes[a].values.begin(), mAttributes[a].values.end(), value);

            // add value to the attribute if it is new (avoid duplicates)
            if (itFind == mAttributes[a].values.end())
            {
                mAttributes[a].values.push_back(value);
            }
        }

        // repeat the process with the ouput class
        const std::string & output = mDataset[i].second;
        const auto          itFind = std::find(mOutputClasses.begin(), mOutputClasses.end(), output);

        // add value to the output classes if it is new (avoid duplicates)
        if (itFind == mOutputClasses.end())
        {
            mOutputClasses.push_back(output);
        }
    }
}

// Node constructor, simply initializes the data
DecisionTree::DecisionTreeNode::DecisionTreeNode(const Dataset & d, const DecisionTreeNode * p) :
    parent(p),
    data(d),
    attributeIndex(-1)
{
}

// recursive tree generation
DecisionTree::DecisionTreeNode * DecisionTree::GenerateTree(const Dataset & data, const DecisionTreeNode * parent, int depth)
{
    // create a new node with the data (will select split axis and value)
    DecisionTreeNode * node = new DecisionTreeNode(data, parent);

    node->attributeIndex = SelectAttribute(data);

    // attribute was not selected, need to make this a leaf node with output class
    if (node->attributeIndex == -1 || depth == 0)
    {
        node->outputClassIndex = SelectOutputClass(node);
        return node;
    }

    // split dataset (one dataset per value of the attribute)
    std::vector<Dataset> splittedDataset = SplitDataset(data, node->attributeIndex);

    // recursively generate tree
    for (unsigned i = 0; i < mAttributes[node->attributeIndex].values.size(); i++)
    {
        node->children.push_back(GenerateTree(splittedDataset[i], node, depth - 1));
    }

    return node;
}

// clean up of all the nodes in the tree
void DecisionTree::FreeTree(DecisionTreeNode * node)
{
    // exit when no node
    if (node == nullptr)
        return;

    // make sure all branches are empty
    for (unsigned i = 0; i < node->children.size(); ++i)
    {
        FreeTree(node->children[i]);
    }

    // delete this specific node of the tree
    delete node;
}

// split given dataset based on the selected attribute (one dataset per value of the attribute)
std::vector<Dataset> DecisionTree::SplitDataset(const Dataset & data, int attributeIndex) const
{
    std::vector<Dataset> splittedDataset(mAttributes[attributeIndex].values.size());

    // traverse the dataset to split based on attribute value
    for (unsigned i = 0; i < data.size(); i++)
    {
        // check the value this sample belongs to
        for (unsigned a = 0; a < mAttributes[attributeIndex].values.size(); a++)
        {
            // once the value of the attribute is matched add it to the correspoding splitted dataset
            if (data[i].first[attributeIndex] == mAttributes[attributeIndex].values[a])
            {
                splittedDataset[a].push_back(data[i]);
                break;
            }
        }
    }

    return splittedDataset;
}

// selects and attribute based on the information gain (ID3) of the data
int DecisionTree::SelectAttribute(const Dataset & data)
{
    int    selectedAttribute = -1;

    std::vector<std::pair<Attribute, bool>> possibleAttributes;
    //check break conditions
    if (CheckEndConditions(data, possibleAttributes) == true) return -1;


    unsigned chosenOutput = ChooseOutputValue(data);
    //compute the entropy of the dataset
    unsigned numberOfYes = 0u;
    for (size_t i = 0u; i < data.size(); i++) {
        if (data[i].second == mOutputClasses[chosenOutput])
            numberOfYes ++;
    }

    double entropy = ComputeEntropy(numberOfYes, data.size());

    std::vector<double> computedInformationGain(possibleAttributes.size());
    //iterate through attributes
    for (size_t att = 0u; att < mAttributes.size(); att++) {
        if (possibleAttributes[att].second == true) continue; //already selected

        double computedValue = 0.;
        for (size_t i = 0u; i < mAttributes[att].values.size(); i++) {
            const std::string& itAttribute = mAttributes[att].values[i];
            unsigned quantityOfValues = 0u;
            unsigned valuesWithOutputTrue = 0u;
            //traverse dataset
            for (size_t dataset_idx = 0u; dataset_idx < data.size(); dataset_idx++) {
                if (itAttribute == data[dataset_idx].first[att]) {
                    quantityOfValues++;
                    //add one to the counter of chosen output
                    if (data[dataset_idx].second == mOutputClasses[chosenOutput])
                        valuesWithOutputTrue++;
                }
            }

            //compute entropy for this attribute
            double ig = ComputeEntropy(valuesWithOutputTrue, quantityOfValues);
            //multiply by the number of this values in the dataset
            computedValue += ig * (static_cast<double>(quantityOfValues) / data.size());
        }
        computedInformationGain[att] = entropy - computedValue;
    }

    double maxInfoGain = -std::numeric_limits<double>().max();
    //return the attribute with bigger information win
    for (size_t i = 0u; i < computedInformationGain.size(); i++) {
        if (computedInformationGain[i] > maxInfoGain) {
            selectedAttribute = i;
            maxInfoGain = computedInformationGain[i];
        }
    }

    if (maxInfoGain == 0.)
        return -1;

    return selectedAttribute;
}

// selects the most probable output class on the given node (leaf node)
int DecisionTree::SelectOutputClass(const DecisionTreeNode * node)
{
    int outputClassIndex    = -1;

    // No samples in the subset
    //Pick the most common class in the parent subset
    if (node->data.empty()) return SelectOutputClass(node->parent);

    //Select the class every sample belongs to
    for (unsigned out_idx = 0u; out_idx < mOutputClasses.size(); out_idx++) {
        bool allSameOutput = true;
        for (unsigned data_idx = 0u; data_idx < node->data.size(); data_idx++) {
            if (node->data[data_idx].second != mOutputClasses[out_idx])
                allSameOutput = false;
        }
        if (allSameOutput)
            return out_idx;
    }

    unsigned nodeDepth = 0u;
    const DecisionTreeNode* n = node->parent;
    while (n) {
        n = n->parent;
        nodeDepth++;
    }
    //Pick the most common output class in the subset
    if (nodeDepth == mMaxDepth) {
        std::vector<unsigned> quantityOfEachOutput(mOutputClasses.size());
        for (auto& it : node->data) {
            for(unsigned out_idx = 0u; out_idx < mOutputClasses.size(); out_idx++)
                if (mOutputClasses[out_idx] == it.second) {
                    quantityOfEachOutput[out_idx]++;
                    break;
                }
        }

        unsigned maxOutputValue = 0u;
        for (unsigned i = 0u; i < quantityOfEachOutput.size(); i++) {
            if (quantityOfEachOutput[i] > maxOutputValue) {
                maxOutputValue = quantityOfEachOutput[i];
                outputClassIndex = i;
            }
        }
    }


    return outputClassIndex;
}

double DecisionTree::ComputeEntropy(const unsigned yes_values, const unsigned total_values) {
    double ig = 0.;
    //compute the IG
    const double d_quantityOfValues = static_cast<double>(total_values);
    const double d_valuesWithOutputTrue = static_cast<double>(yes_values);
    if (yes_values == 0u)
        ig = 0.;
    else if (yes_values == total_values)
        ig = 0.;
    else {

        ig = d_valuesWithOutputTrue / d_quantityOfValues *
            log2(d_valuesWithOutputTrue / d_quantityOfValues) +
            (d_quantityOfValues - d_valuesWithOutputTrue) / d_quantityOfValues *
            log2((double)(d_quantityOfValues - d_valuesWithOutputTrue) / d_quantityOfValues);
        ig *= -1.;
    }
    return ig;
}

unsigned DecisionTree::ChooseOutputValue(const Dataset& data) {
    if (mOutputClasses.size() == 2)  return 1u;

    std::vector<unsigned> quantityOfEachOutput(mOutputClasses.size());
    for (auto& it : data) {
        for (unsigned out_idx = 0u; out_idx < mOutputClasses.size(); out_idx++)
            if (mOutputClasses[out_idx] == it.second) {
                quantityOfEachOutput[out_idx]++;
                break;
            }
    }

    unsigned index = 0u;
    unsigned maxOutputValue = 0u;
    for (unsigned i = 0u; i < quantityOfEachOutput.size(); i++) {
        if (quantityOfEachOutput[i] > maxOutputValue) {
            maxOutputValue = quantityOfEachOutput[i];
            index = i;
        }
    }
    return index;
}

bool DecisionTree::CheckEndConditions(const Dataset& data, std::vector<std::pair<Attribute, bool>>& possibleAttributes) {
    //No samples in the subset
    if (data.empty()) return true;

    //Every sample in the subset belongs to the same class 
    bool allSameOutput = true;
    for (unsigned dataset_idx = 1u; dataset_idx < data.size(); dataset_idx++) {
        if (data[dataset_idx].second != data[dataset_idx - 1].second) {
            allSameOutput = false;
            break;
        }
    }
    if (allSameOutput)
        return true;
   
    for (unsigned i = 0u; i < mAttributes.size(); i++) {
        bool allSameValues = true;
        for (unsigned dataset_idx = 1u; dataset_idx < data.size(); dataset_idx++) {
            if (data[dataset_idx].first[i] != data[dataset_idx - 1].first[i]) {
                allSameValues = false;
                break;
            }
        }
        possibleAttributes.push_back({ mAttributes[i], allSameValues });
    }


    return false;
}
} // namespace CS397