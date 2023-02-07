/*-------------------------------------------------------
File Name: IsolationForest.cpp
Author: Unai Morentin
---------------------------------------------------------*/

#include "IsolationForest.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace CS397
{

// constructor will create the isolation forest for the given data
IsolationForest::IsolationForest(const Dataset & data, unsigned numTrees) :
    mDataset(data),
    mTreeRoots(numTrees, nullptr)
{
    // generate the forest by creating the trees
    for (unsigned i = 0; i < numTrees; i++)
    {
        mTreeRoots[i] = GenerateTree(data);
    }
}

// destructor will free all the memory of the trees
IsolationForest::~IsolationForest()
{
    // free every tree in the forest
    for (unsigned i = 0; i < mTreeRoots.size(); i++)
    {
        FreeTree(mTreeRoots[i]);
    }
}

// recursive tree generation
IsolationForest::iTreeNode * IsolationForest::GenerateTree(const Dataset & data)
{
    if (data.size() == 0u)
        return nullptr;

    iTreeNode* newNode = new iTreeNode(data);

    //exit condition
    if (data.size() == 1u)
        return newNode;

    const unsigned& splitAxis = newNode->splitAxis;

    Dataset data_left, data_right;
    //clasify data depending on the split axis and value
    for (unsigned data_idx = 0u; data_idx < data.size(); data_idx++) {
        //smaller or equal value goes left
        if (data[data_idx][splitAxis] < newNode->splitValue) {
            data_left.push_back(data[data_idx]);
        }
        //goes right
        else {
            data_right.push_back(data[data_idx]);
        }
    }

    //recursive call
    newNode->left = GenerateTree(data_left);
    newNode->right = GenerateTree(data_right);

    return newNode; //return root
}

// clean up of all the nodes in the tree
void IsolationForest::FreeTree(iTreeNode * node)
{
    // exit when no node
    if (node == nullptr)
        return;

    // make sure both branches are empty
    FreeTree(node->left);
    FreeTree(node->right);

    // delete this specific node of the tree
    delete node;
}

// isolation tree node constructor (selects random split axis and value)
IsolationForest::iTreeNode::iTreeNode(const Dataset & d) :
    left(nullptr),
    right(nullptr),
    splitAxis(0),
    splitValue(0.0)
{
    assert(d.size() > 0);
    splitAxis = static_cast<unsigned int>(rand()) % d.front().size();
    double min = std::numeric_limits<double>().max();
    double max = -std::numeric_limits<double>().max();

    for (unsigned i = 0u; i < d.size(); i++) {
        if (d[i][splitAxis] < min)
            min = d[i][splitAxis];
        if (d[i][splitAxis] > max)
            max = d[i][splitAxis];
    }

    const double r = static_cast<double>(rand()) / RAND_MAX;
    splitValue = min + r * (max - min);
}

// computes the path length for the given datapoint (input) on the given node recursively
unsigned IsolationForest::PathLength(const iTreeNode * node, const std::vector<double> & input)
{
    unsigned pathLength = 0u;
    const iTreeNode* n = node;

    while (n) {
        if (input[n->splitAxis] < n->splitValue)
            n = n->left;
        else
            n = n->right;
        pathLength++;
    }

    return pathLength;
}

// returns the anomaly score of the given input for the create isolation forest
double IsolationForest::ComputeScore(const std::vector<double> & input) const
{
    // compute the average path length in every tree for the given input
    double avgPathLength = 0.0;
    for (unsigned i = 0; i < mTreeRoots.size(); i++)
    {
        avgPathLength += PathLength(mTreeRoots[i], input);
    }
    avgPathLength /= mTreeRoots.size();

    // compute c, the average traversal path for unsuccesfull search
    const double EulerMascheronitConstant = 0.5772156649;
    const double H                        = std::log(mDataset.size() - 1.0) + EulerMascheronitConstant;
    const double c                        = 2.0 * H - (2.0 * mDataset.size() - 1.0) / mDataset.size();

    // use Isolation forest formula for anomaly score
    const double score = std::pow(2.0, -avgPathLength / c);

    return score;
}

} // namespace CS397