# ExpressionRecognition

##Introduction
The objective of this project is to guess facial expressions of a human face.
The emotions that will be taken in consideration are: 
 * Neutral, Hapiness, Sadness, Surprise, Anger, Disgust and Fear. 

These are amongst the most relevant emotions, however some were left behind (such as comtempt, envy, boredom, etc.), mainly because a dataset for those wasn't obtained.


##Strategies
The first stage is to detect all the human faces in the image. It is achieved using a pre-trained predictor from OpenCV.
Then, knowing the rectangular areas of the image that contain faces, a model is used to predict the facial expression.
Two models where developed to answer this problem, which use different techniques to aproach the problem:
 * The fisherfaces classifier
 * Detection of facial landmark points and posterior classification using a Support Vector Machine (SVM). 



##Usage
The following libraries are needed to run the program: numpy, OpenCV and dlib.
To run use: 

`python expression_rec.py [option]`
 
