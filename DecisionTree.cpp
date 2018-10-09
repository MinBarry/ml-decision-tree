
#include "DecisionTree.h"

DecisionTree::DecisionTree(int ** train_data,int ** test_data,int attrNum, int instanceNum, int testInstan)
{
    train_set = train_data;
    test_set = test_data;
    ATTR_NUM = attrNum;
    INSTANCES_NUM = instanceNum;
    TEST_INSTANCES_NUM = testInstan;
    root = new struct Node;
}

DecisionTree::~DecisionTree()
{
    //dtor

}

void DecisionTree::stump(){
    std::vector<int>* set = new std::vector<int>;
    for(int i=0; i<INSTANCES_NUM;i++){
        set->push_back(i);
    }
    int attrIndex, value;
    BestSplit(set,attrIndex,value);
    //create children nodes
    struct Node * rightChild, * leftChild;
    rightChild = new struct Node;
    leftChild = new struct Node;

    root->rightChild = rightChild;
    root->leftChild = leftChild;
    root->exampleSet= set;
    root->attrNum = attrIndex;
    root->attrValue = value;
    root->clazz = -1;

    std::vector<int> *posSub = new std::vector<int>;
    std::vector<int> *negSub = new std::vector<int>;
    Split(set,attrIndex,value, posSub,negSub);

    root->gain = SplitBenefit(set,posSub,negSub);

    rightChild->exampleSet = posSub;
    rightChild->rightChild =0;
    rightChild->leftChild =0;
    leftChild->exampleSet = negSub;
    leftChild->rightChild =0;
    leftChild->leftChild =0;

    if(Entropy(posSub)==0){
        //rightChild is a leaf
        rightChild->attrNum = -1;
        rightChild->attrValue = -1;
        rightChild->clazz = train_set[posSub->at(0)][0];
        rightChild->gain=-1;

    }else{
        BestSplit(posSub,attrIndex,value);
        rightChild->attrNum = attrIndex;
        rightChild->attrValue = value;
        rightChild->clazz = -1;
        rightChild->gain=-1;
    }
    if(Entropy(negSub)==0){
        leftChild->attrNum = -1;
        leftChild->attrValue = -1;
        leftChild->clazz = train_set[negSub->at(0)][0];
        leftChild->gain=-1;
    }else{
        BestSplit(negSub,attrIndex,value);
        leftChild->attrNum = attrIndex;
        leftChild->attrValue = value;
        leftChild->clazz = -1;
        leftChild->gain=-1;
    }

}
void DecisionTree::GenerateDT(){
    std::vector<int>* set = new std::vector<int>;
    for(int i=0; i<INSTANCES_NUM;i++){
        set->push_back(i);
    }
    expandTree(root,set);
}
void DecisionTree::expandTree(struct Node * root,std::vector<int>* set){
    int attrIndex, value;
    //get the attribute and value that results in the best split
    BestSplit(set,attrIndex,value);
    //create children nodes
    struct Node * rightChild, * leftChild;
    rightChild = new struct Node;
    leftChild = new struct Node;
    //set up current tree node's values
    root->rightChild = rightChild;
    root->leftChild = leftChild;
    root->exampleSet= set;
    root->attrNum = attrIndex;
    root->attrValue = value;
    root->clazz = -1;

    std::vector<int> *posSub = new std::vector<int>;
    std::vector<int> *negSub = new std::vector<int>;
    //split current tree node's set to positive and negative sets based on best split attribute
    Split(set,attrIndex,value, posSub,negSub);
    root->gain = SplitBenefit(set,posSub,negSub);
    //for each child check if corresponding sets are pure
    //create a leaf if set was pure
    if(Entropy(posSub)==0){
        //rightChild is a leaf
        rightChild->exampleSet = posSub;
        rightChild->rightChild =0;
        rightChild->leftChild =0;
        rightChild->attrNum = -1;
        rightChild->attrValue = -1;
        rightChild->clazz = train_set[posSub->at(0)][0];
        rightChild->gain=-1;

    } else{
        //expand right child if positive set was not pure
        expandTree(rightChild,posSub);
    }
    if(Entropy(negSub)==0){
        leftChild->exampleSet = negSub;
        leftChild->rightChild =0;
        leftChild->leftChild =0;
        leftChild->attrNum = -1;
        leftChild->attrValue = -1;
        leftChild->clazz = train_set[negSub->at(0)][0];
        leftChild->gain=-1;
    }
    else{
        //expand left child of negative set was not pure
        expandTree(leftChild, negSub);
    }

}

void DecisionTree::BestSplit(std::vector<int>* set, int& attrIndex, int& value){
    double benefit=-1;
    double maxbenefit=-1;
    int bestAttr,bestVal;
    //first col in dataset (column 0) is the class value not an attribute
    for(int i=1; i<=ATTR_NUM;i++){
        std::vector<int> numValues = getAttrValues(set,i);
        while(!numValues.empty()){
            std::vector<int>* posSub = new std::vector<int>;
            std::vector<int>* negSub = new std::vector<int>;

            int tempVal = numValues.back();
            numValues.pop_back();

            Split(set,i,tempVal, posSub,negSub);
            if(posSub->empty() || negSub->empty()) continue;
            benefit = SplitBenefit(set,posSub,negSub);
            if(benefit>maxbenefit){
                maxbenefit=benefit;
                bestAttr=i;
                bestVal=tempVal;
            }
        }
    }
    attrIndex = bestAttr;
    value = bestVal;
}

std::vector<int> DecisionTree::getAttrValues(std::vector<int>* set,int attrIndex){
    std::vector<int> values;
    for(int i=0; i< set->size(); i++){
        bool add = true;
        for(std::vector<int>::iterator itr = values.begin(); itr!=values.end(); ++itr){
            if(train_set[set->at(i)][attrIndex] == *itr){
                add=false;
                break;
            }
        }
        if(add){

            values.push_back(train_set[set->at(i)][attrIndex]);
        }
    }
    return values;
}

void DecisionTree::Split(std::vector<int>* set,int attrIndex,int attrVal, std::vector<int>* posSub,std::vector<int>* negSub){
    for(int i=0; i< set->size(); i++){
        if(train_set[set->at(i)][attrIndex] == attrVal){
            posSub->push_back(set->at(i));
        }else{
            negSub->push_back(set->at(i));
        }
    }
}
double DecisionTree::SplitBenefit(std::vector<int>* set,std::vector<int>* posSub,std::vector<int>*negSub){
    double p1 = (double)posSub->size()/(double)set->size();
    double p2 = (double)negSub->size()/(double)set->size();
    double s = Entropy(set);
    double s1 = p1* Entropy(posSub);
    double s2 = p2* Entropy(negSub);
    double benefit = s - s1 - s2;
    return benefit;
}

double DecisionTree::Entropy(std::vector<int>* set){
    double pos= CountPositiveInstances(set);
    double neg = set->size()-pos;
    double h1,h2;
    if(pos !=0){
        h1 = -(pos/(double)set->size())* log2(pos/(double)set->size());
    }else {
        h1 = 0;
    }
    if(neg !=0){
       h2 = -(neg/(double)set->size())* log2(neg/(double)set->size());
    }else{
        h2 = 0;
    }
    return h1 + h2;

}

int DecisionTree::CountPositiveInstances(std::vector<int>* set){
    int count =0;
    for(int i=0; i<set->size();i++){
        if(train_set[set->at(i)][0]==1){
            count++;
        }
    }
    return count;
}

void DecisionTree::printLevelOrder() {
  if (!root) return;
  std::queue<Node*> nodesQueue;
  int nodesInCurrentLevel = 1;
  int nodesInNextLevel = 0;
  nodesQueue.push(root);
  while (!nodesQueue.empty()) {
    Node *currNode = nodesQueue.front();
    nodesQueue.pop();
    nodesInCurrentLevel--;
    if (currNode) {
		if((currNode->attrNum)!=-1){
			std::cout <<"x"<< currNode->attrNum<<" : "<<currNode->attrValue <<" | ("<<currNode->gain<<")";}
		else{
			std::cout <<" ~ "<< currNode->clazz <<" ~ ";
		}
      nodesQueue.push(currNode->leftChild);
      nodesQueue.push(currNode->rightChild);
      nodesInNextLevel += 2;
    }
    if (nodesInCurrentLevel == 0) {
      std::cout << "\n";
      nodesInCurrentLevel = nodesInNextLevel;
      nodesInNextLevel = 0;
    }
  }
}

int DecisionTree::predict(int * row){
    struct Node * temp = root;

    while(1){
        //if temp is not a leaf
        if(temp->rightChild!=0 && temp->attrValue == row[temp->attrNum]){
            //move to right child
            temp = temp->rightChild;
        }else if(temp->leftChild!=0 && temp->attrValue != row[temp->attrNum]){
            temp = temp->leftChild;
        }else {
            break;
        }
    }
    return determineClass(temp);
}

double DecisionTree::trainError(){
    double error=0;
    for(int i=0; i< INSTANCES_NUM; i++){
        int pred = predict(train_set[i]);
        if(pred != train_set[i][0]){
            error++;
        }
    }
    return error/INSTANCES_NUM;
}

double DecisionTree::testError(){
    double error=0;
    for(int i=0; i< TEST_INSTANCES_NUM; i++){
        int pred = predict(test_set[i]);
        if(pred != test_set[i][0]){
            error++;
        }
    }
    return error/TEST_INSTANCES_NUM;
}

int DecisionTree::determineClass(struct Node * node){

    if(node->clazz != -1) return node->clazz;

    int pos=0, neg=0;
    for(int i=0; i< node->exampleSet->size();i++){
        if(train_set[node->exampleSet->at(i)][0]==1){
            pos++;
        }else{
            neg++;
        }
    }
    if(pos>neg) return 1;
    else return 0;

}

