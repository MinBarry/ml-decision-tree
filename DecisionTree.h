/**
* Gilderlane Alexandre
* Minna Barry
* CS434 - HW3 - Decision Tree
**/
#include <vector>
#include <cmath>
#include <queue>
#include <iomanip>
#include <iostream>

#ifndef DECISIONTREE_H
#define DECISIONTREE_H


class DecisionTree
{
    public:
        DecisionTree(int ** train_data,int ** test_data,int attrNum, int instanceNum, int testInstan);
        virtual ~DecisionTree();

        int ATTR_NUM;
        int INSTANCES_NUM;          //train set instances number
        int TEST_INSTANCES_NUM;
        int ** train_set;
        int ** test_set;

        //represents a tree node
        struct Node{
            //a subset of train_set with values of the attrNum attribute that are equal to attrValue
            std::vector<int>* exampleSet;
            //class of a leaf node. = -1 if node is not a leaf
            int clazz;
            double gain;
            int attrNum;
            int attrValue;
            //pointers to children
            struct Node * rightChild;
            struct Node * leftChild;
        };

        struct Node * root;

        //learns full tree
        void GenerateDT();
        //learns tree stump
        void stump();
        //prints tree in level order
        void printLevelOrder();
        //returns a class for the passed row
        int predict(int * row);
        //calculates the train error rate
        double trainError();
        //calculates the test error rate
        double testError();


    protected:
    private:
        //returns the attribute number and value that makes the best split of the passed set
        void BestSplit(std::vector<int>* set, int& attrIndex, int& value);
        //returns the number of values a given attribute has
        std::vector<int> getAttrValues(std::vector<int>* set,int attrIndex);
        //splits set to two subsets based on passed attribute value
        void Split(std::vector<int>* set,int attrIndex,int attrVal, std::vector<int>* posSub,std::vector<int>* negSub);
        //calculates the benefit from splitting data to posSub and negSub
        double SplitBenefit(std::vector<int>* set,std::vector<int>* posSub,std::vector<int>* negSub);
        //calculates the entropy of the passed set
        double Entropy(std::vector<int>* set);
        //return the number of positive entries in the passed set
        int CountPositiveInstances(std::vector<int>* set);
        //returns the majority class if the passed node was not a leaf
        int determineClass(struct Node * node);
        //used by generateDT to expand the tree's nodes
        void expandTree(struct Node * root,std::vector<int>* set);


};

#endif // DECISIONTREE_H
