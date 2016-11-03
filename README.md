# Object-Detection
A project that detects specific objects in images/videos using an SVM on HOG features. The program is written in C++.

# Source programs
* hogtrainfeatures.cpp : Extracts HOG features from a given set of images and stores them in a .xml file.
* hogtrainsvm.cpp : Uses the extracted HOG features and trains a SVM with them.
* hogobjectdetect.cpp : Uses the trained SVM to detect objects in images/livefeed/videos.

# Dependencies
OpenCV (used version 3.0.1)

# Trial
Code to be run in terminal

```ruby
$ cd Object-Detection
$ make train
$ ./output /PathToDirectoryContainingPositiveImages/ positive.xml
$ ./output /PathToDirectoryContainingNegativeImages/ negative.xml
$ make svm
$ ./output positive.xml negative.xml trainedsvm.xml
$ make all
# for detection in image
$ ./output trainedsvm.xml /PathToTestImage
# for detection in livefeed
$ ./output trainedsvm.xml
# for detection in video
$ ./output trainedsvm.xml /PathToTestVideo
```
# Sample outputs

* PepsiCan


![detectedpepsican2](https://cloud.githubusercontent.com/assets/17588365/19882571/9647c022-a034-11e6-9ab0-4f31e5ecfa33.png)



* Leaf


![detectionleaf6](https://cloud.githubusercontent.com/assets/17588365/19882572/9d3f615a-a034-11e6-98da-a32b08d01eb3.png)


