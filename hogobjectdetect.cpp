//detect objects using a SVM trained using hog features from positive and negative images of concerned objects

//#include "types.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
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
#include <fstream>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

int help();

void svmdetector(Ptr<SVM>& svm,vector< float >& hogdetector);

void drawlocations(Mat& draw,const vector< Rect >& objectlocations);//,const vector< Point >& foundpoints);

int main(int argc,char** argv)
{
     cout<<"argc : "<<argc<<endl;
     int option=help();

     cout<<"Declaring variables.."<<endl;
     int esc=27;
     Mat img,grayimg,draw;
     vector< Rect > objectlocations;
     vector< Point > foundpoints;
     string trainedsvmname=argv[1];

     cout<<"Loading trained SVM file.."<<endl;
     //Load trained SVM xml file
     Ptr<SVM> svm=SVM::load(argv[1]);

     cout<<"Declaring HOG descriptor.."<<endl;
     HOGDescriptor hog(Size(64,48),Size(8,8),Size(4,4),Size(4,4),9);
     //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//note : winsize must be smaller than image size
     //hog.winSize=Size(16,12);
     //HOGDescriptor hog(Size(32,24),Size(16,16),Size(8,8),Size(8,8),9);//note : winsize must be smaller than image size

     cout<<"Calling svmdetector.."<<endl;
     vector < float >  hogdetector;
     svmdetector(svm,hogdetector);

     cout<<"Back to main.."<<endl;

     cout<<"Size of hogdetector after coming back to main : "<<sizeof(hogdetector)<<endl;
     hog.setSVMDetector(hogdetector);
     cout<<"Size of hogdetector after 'setSVMDetector' : "<<sizeof(hogdetector)<<endl;
     cout<<"Size of hog after 'setSVMDetector' : "<<sizeof(hog)<<endl;

     cout<<"Option is : "<<option<<endl;

     if((option!=1)&&(option!=2)&&(option!=3))
     {
      cout<<"First if\tOption is : "<<option<<endl;
      exit(0);
     }

     if(option==3)
     {
          cout<<"Option is : "<<option<<endl;

          cout<<"argc : "<<argc<<endl;

          if(argc!=3)
          { 
           cout<<"Type the path to the video as the third argument in the command line"<<endl; 
           return -1; 
          }
          cout<<"Opening video now.."<<endl;
          VideoCapture vdo(argv[2]);
          while(1)
          {
               bool imgread=vdo.read(img);
               if(!imgread) 
               {
                cout<<"Video not opening";
                return -1;
               }
               resize(img,img,Size(256,192));//(128,96)
               //imshow("Original",img);

               draw=img.clone();
               //cout<<"Cloned image.."<<endl;

               objectlocations.clear();

               cout<<"Calling detectMultiScale.."<<endl;

               cout<<"Size of objectlocations before calling hog.detectMultiScale"<<objectlocations.size()<<endl;

               cout<<"Size of hog before detectMultiScale : "<<sizeof(hog)<<endl;
 
               hog.detectMultiScale(img,objectlocations,0.0,Size(9,9),Size(5,5),1.05,1.5);//(9,9),(5,5)
               
               cout<<"Size of objectlocations after calling hog.detectMultiScale"<<objectlocations.size()<<endl;
               cout<<"Calling drawlocations.."<<endl;
               drawlocations(draw,objectlocations);//,foundpoints);
           
               //show detected objects
               imshow("Detected",draw);
               if(waitKey(30)==esc) break;          
          }
     }

     if(option==2)
     {
      cout<<"Option is : "<<option;//<<endl;
       //starting webcam
      cout<<endl<<"Starting webcam now.."<<endl;
      VideoCapture vdo(0);
      while(1)
      {
           bool imgread=vdo.read(img);
           if(!imgread) 
           {
            cout<<"Cam not opening";
            return -1;
           }
           imshow("LiveFeed",img);
 
           /*draw=img.clone();
           objectlocations.clear();
 
           hog.detectMultiScale(img,objectlocations);
           drawlocations(draw,objectlocations);
 
           objectlocations.clear();
           
           //show detected objects
           imshow("Detected",draw);
       */
           if(waitKey(30)==esc) break;
      }
     }

     if(option==1)
     {
          cout<<"Option is : "<<option<<endl;

          cout<<"argc : "<<argc<<endl;

          if(argc!=3)
          { 
           cout<<"Type the path to the image as the third argument in the command line"<<endl; 
           return -1; 
          }

          img=imread(argv[2]); 
          resize(img,img,Size(128,96));//(128,96)
          imshow("Original",img);

          draw=img.clone();
          cout<<"Cloned image.."<<endl;

          objectlocations.clear();

          cout<<"Calling detectMultiScale.."<<endl;

          cout<<"Size of objectlocations before calling hog.detectMultiScale"<<objectlocations.size()<<endl;

          cout<<"Size of hog before detectMultiScale : "<<sizeof(hog)<<endl;
 
          hog.detectMultiScale(img,objectlocations,0.0,Size(9,9),Size(5,5),1.05,1.5);//(9,9),(5,5)
          //hog.detectMultiScale(img,objectlocations,0.0,Size(9,9),Size(),1.05,2);//,true);//(9,9),(5,5)
          //cout<<"winStride after calling hog.detectMultiScale() : "<<hog.winStride<<endl; 
          //second-last parameter 'scale' makes virtually no difference in the detection
          //third parameter 'hit-threshold'" " " "
          //hog.detectMultiScale(img,objectlocations,0.0,Size(4,4),Size(),1.01,0.1);

          //hog.detectMultiScale(img,objectlocations);
          //hog.detect(img,foundpoints);
          //cout<<"Size of Point after hog.detect : "<<sizeof(Point)<<endl;
          //cout<<"Size of hog before detectMultiScale : "<<sizeof(hog)<<endl;
          //cout<<objectlocations.size()<<endl;
          cout<<"Size of objectlocations after calling hog.detectMultiScale"<<objectlocations.size()<<endl;

          //cout<<foundpoints[0]->x<<'\t'<<foundpoints[0]->y<<endl;

          cout<<"Calling drawlocations.."<<endl;
          drawlocations(draw,objectlocations);//,foundpoints);

          //objectlocations.clear();
          
          //show detected objects
          imshow("Detected",draw);
      } 

     cout<<"end"<<endl;
     waitKey(0);
     return -1;
}

int help()
{
     int option;
     cout<<endl<<"Press 1 to detect object in an image, 2 to detect in a live feed using webcam and 3 to detect in a video : ";
     cin>>option;
     return option;
}

void svmdetector(Ptr<SVM>& svm,vector< float >& hogdetector)
{
    cout<<"Getting support vectors.."<<endl;
    //getting support vectors
     Mat sv=svm->getSupportVectors();
     int svtot=sv.rows;
     cout<<endl<<"Number of rows and columns respectively in SVM :"<<svtot<<'\t'<<sv.cols<<endl;
     Mat alpha,svidx;
     double rho=svm->getDecisionFunction(0,alpha,svidx);

     CV_Assert(alpha.total()==1 && svidx.total()==1 && svtot==1);
     CV_Assert((alpha.type()==CV_64F && alpha.at<double>(0)==1.) || (alpha.type()==CV_32F && alpha.at<float>(0)==1.f));
     CV_Assert(sv.type()==CV_32F);

     hogdetector.clear();
     hogdetector.resize(sv.cols+1);
     memcpy(&hogdetector[0], sv.ptr(),sv.cols*sizeof(hogdetector[0]));
     hogdetector[sv.cols]=(float)-rho;
     cout<<"Size of hogdetector : "<<sizeof(hogdetector)<<endl;
}

void drawlocations(Mat& draw,const vector< Rect >& objectlocations)//,const vector< Point >& foundpoints)
{
     cout<<"Inside draw function.."<<endl;
     if(objectlocations.empty()) exit(0);

     cout<<"Inside draw function.."<<endl;
     imshow("In draw function",draw);
 
     RNG rng(12345);

     vector< Rect >::const_iterator start   =  objectlocations.begin();
     vector< Rect >::const_iterator end     =  objectlocations.end();
     int i;
     for( ; start!=end ; start++)
          rectangle(draw,*start,Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ),1);
     cout<<objectlocations.size();

     /*for(int i=0;i<objectlocations.size();i++)
          rectangle(draw,objectlocations[i],Scalar(0,255,0),5);*/
     //cvtColor(draw,draw,CV_GRAY2BGR);
     //resize(draw,draw,Size(400,400));
     imshow("Detectedfirst",draw);

}
