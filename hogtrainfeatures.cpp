//train HOG
#include <dirent.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <vector>
#include <opencv2/video/video.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <string.h>

using namespace cv;
using namespace std;

int main(int argc,char** argv)
{
     DIR *d;
     string strprefix=argv[1];
     string imagename;
     string HogFeaturesFileName=argv[2];
     vector< vector <float> > vec_descriptorvalues;
     vector< vector <Point> > vec_locations;

     Mat image;
     int i;

     struct dirent *dir;
     d=opendir(argv[1]);

     if(d)
     {
      while((dir=readdir(d))!=NULL)
      {
           imagename=strprefix+dir->d_name;
           image=imread(imagename,1);
           if(!image.empty())
           {
            cvtColor(image,image,CV_RGB2GRAY);
            resize(image,image,Size(64,48));
            //extract features
            //HOGDescriptor h(Size(32,16),Size(8,8),Size(4,4),Size(4,4),9);
            HOGDescriptor h(Size(64,48),Size(8,8),Size(4,4),Size(4,4),9);
            vector<float> descriptorvalues;
            vector<Point> locations;
            h.compute(image,descriptorvalues,Size(0,0),Size(0,0),locations);
            vec_descriptorvalues.push_back(descriptorvalues);
            vec_locations.push_back(locations);       
           }
           waitKey(0);
      }
      closedir(d);
     }

     //saving to xml
     FileStorage hogxml(HogFeaturesFileName,FileStorage::WRITE);
     int row=vec_descriptorvalues.size();
     int col=vec_descriptorvalues[0].size();
     Mat M(row,col,CV_32F);
     //save Mat to xml
     for(i=0;i<row;i++)
          memcpy(&(M.data[col*i*sizeof(float) ]),vec_descriptorvalues[i].data(),col*sizeof(float));
     //write xml
     write(hogxml,"DescriptorOfImages",M);
     hogxml.release();
     waitKey(0);
     return -1;
}




