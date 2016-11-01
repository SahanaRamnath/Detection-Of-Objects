# the compiler
CC = g++

# the compiler flags
CFLAGS = `pkg-config opencv --cflags`
LIBS = `pkg-config opencv --libs` -I.

all: objectdetect

objectdetect: hogobjectdetect.cpp
	$(CC) $(CFLAGS) -o output hogobjectdetect.cpp $(LIBS)

train: hogtrainfeatures.cpp
	$(CC) $(CFLAGS) -o output hogtrainfeatures.cpp $(LIBS)

svm: hogtrainsvm.cpp
	$(CC) $(CFLAGS) -o output hogtrainsvm.cpp $(LIBS)
