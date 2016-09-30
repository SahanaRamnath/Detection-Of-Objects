#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <stdarg.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml/ml.hpp"

void help();

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc,char** argv)
{
     if(argc!=4)
     {  
          help();
          return -1;
     }

     //read HOG features from .xml file
     cout<<endl<<"Feature data loading.."<<endl;
     FileStorage readpositivexml(argv[1],FileStorage::READ);
     FileStorage readnegativexml(argv[2],FileStorage::READ);

     //positive Mat
     Mat pMat;
     readpositivexml["DescriptorOfImages"]>>pMat;
     int pRow=pMat.rows,pCol=pMat.cols;

     //negative Mat
     Mat nMat;
     readnegativexml["DescriptorOfImages"]>>nMat;
     int nRow=nMat.rows,nCol=nMat.cols;

     readpositivexml.release();
     readnegativexml.release();

     cout<<"Making training data for SVM.."<<endl;
     //descriptor data set
     Mat PNDescriptor(pRow+nRow,pCol,CV_32FC1);
     memcpy(PNDescriptor.data,pMat.data,sizeof(float)*pCol*pRow);     

     int startP=sizeof(float)*pMat.cols*pMat.rows;
     memcpy(&(PNDescriptor.data[startP]),nMat.data,sizeof(float)*nCol*nRow);

     //data labeling
     Mat labels(pRow+nRow,1,CV_32SC1,Scalar(-1));
     labels.rowRange(0,pRow)=Scalar(1);

     //set up SVM's parameters and train it
     cout<<"Putting Parameters.."<<endl;   
     Ptr<SVM> svm=SVM::create();
     svm->setType(SVM::C_SVC);//C_SVC
     svm->setCoef0(0.5);
     svm->setDegree(3);
     svm->setGamma(0);
     //svm->setType(SVM::EPS_SVR);
     svm->setKernel(SVM::LINEAR);
     //TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,1000,1e-3);
     TermCriteria criteria(TermCriteria::MAX_ITER,10000,1e-6);
     svm->setTermCriteria(criteria);
     svm->setNu(0.8);
     svm->setP(10);
     svm->setC(0.01);
     //cout<<"svm->C : "<<svm->C<<endl;
     //cout<<"svm->Nu : "<<svm->Nu<<endl;
     //cout<<"svm->Coef0 : "<<svm->Coef0<<endl;
     //cout<<"svm->Degree : "<<svm->Degree<<endl;
     //cout<<"svm->Gamma : "<<svm->Gamma<<endl;
     //cout<<"svm->P : "<<svm->P<<endl;
     cout<<"SVM Training.."<<endl;
     svm->train(PNDescriptor,ROW_SAMPLE,labels);
     //svm->train(PNDescriptor,ROW_SAMPLE,labels);
     //const Ptr<TrainData> svmresult=TrainData::create(PNDescriptor,ROW_SAMPLE,labels);
     //svm->trainAuto(svmresult);
     //Ptr<StatModel> svmstat;
     //bool svmbool=svmstat->StatModel::train(&svmresult,0);
     //bool svmbool=svmstat->StatModel::train(PNDescriptor,ROW_SAMPLE,labels);
     //cout<<svmbool; 

     //saving trained data
     cout<<"Saving SVM xml now.."<<endl;
     //FileStorage svmxml(argv[3],FileStorage::WRITE);
     //svm->write(svmxml);
     //svmxml.release();
     //svmresult->save(argv[3]);
     //svm->StatModel::save(argv[3]);
     svm->save(argv[3]);
     cout<<"Saved"<<endl;
     //svm->release();
     return -1;
}

void help()
{
 cout<<"./output positivehogfeatures.xml negativehogfeatures.xml trainedsvmname.xml"<<endl;
}
