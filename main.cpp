
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "DecisionTree.h"

int ** readCSVData(char * fileName, int rows, int cols);

int ** readCSVData(char * fileName, int rows, int cols){
    std::ifstream inf;
    inf.open(fileName);
    std::stringstream ss;
    std::string s;
    int ** data = new int * [rows];
    for(int i=0; i<rows;i++){
        data[i] = new int [cols];
    }
   for(int i=0;i<rows;i++){
        inf >> s;
        for(int j=0;j<cols;j++){
            std::string temp;
            temp = s.substr(j*2,1);
            data[i][j] = atoi(temp.c_str());
        }
    }
    return data;
}

void print2DArray(int ** arr, int rows,int cols){
   for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            std::cout<<arr[i][j] <<" ";
        }
        std::cout<<"\n";
    }
}
int main(){
    int attributes = 6;
    int instances = 124;
    int ** train_data = readCSVData("monks-1-train.csv",instances,attributes+1);

    int testInstances = 432;
    int ** test_data = readCSVData("monks-1-test.csv",testInstances, attributes+1);

    DecisionTree MonkDT(train_data,test_data,attributes,instances,testInstances);
    MonkDT.GenerateDT();
    MonkDT.printLevelOrder();
    std::cout<<"DT Train Error: "<<MonkDT.trainError();
    std::cout<<"\nDT Test Error: "<<MonkDT.testError()<<"\n\n";

    DecisionTree MonkDS(train_data,test_data,attributes,instances,testInstances);
    MonkDS.stump();
    MonkDS.printLevelOrder();
    std::cout<<"Stump Train Error: "<<MonkDS.trainError();
    std::cout<<"\nStump Test Error: "<<MonkDS.testError()<<"\n\n";



    return 0;
}
