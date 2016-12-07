import cv2
import time
import sys
import dlib
import glob
import random
from skimage import io
import numpy as np

emotions = ["neutral", "happy", "sadness", "anger", "disgust", "surprise", "fear", "contempt"]
fishface = cv2.createFisherFaceRecognizer()

data = {}

#definition of wrapper classes http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''Wrapper for OpenCV SVM algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1)
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float( [self.model.predict(s) for s in samples])

#main----------
def main():
    chosen_method = 0
    if(len(sys.argv) > 1):

        #debug face extration in the dataset and dataset creation
        if(sys.argv[1] == "1"):
            print "debug face extration in the dataset and dataset creation"

            for emotion in emotions:
                check_faces(emotion);

        #Run fisher faces trainer on database
        elif(sys.argv[1] == "2"):
            print "Run fisher faces trainer on database"
            metascore = []
            for i in range(0,10):
                correct , size_training = run_recognizer(len(emotions))
                print "got {} percent correct".format(correct)
                metascore.append(correct)
            print "Mean score: {} percent correct".format(np.mean(metascore))
            '''saving the model obtained'''
            fishface.save('fishface_emotion_detect_model.xml')
            chosen_method = 1

        #load a fisherface model and run webcam
        elif(sys.argv[1] == "3"):
            print "load a fisherface model and run webcam"
            if(len(sys.argv[2]) > 2):
                fishface.load(sys.argv[2])
            else:
                fishface.load('emotion_detection_model.xml')
            chosen_method = 1

        #self train fisherface and run webcam
        elif(sys.argv[1] == "4"):
            print "self train fisherface and run webcam"
            train_fisher_self()
            chosen_method = 1
            fishface.save('fishface_self_emotion_detect_model.xml')

        #test with incremental number of emotions
        elif(sys.argv[1] == "5"):
            print "test with incremental number of emotions"
            incremental_fisher_faces_test()
            chosen_method = -1

    if(chosen_method >= 0):
        #begin webcam capture
        image_capture(chosen_method)

def image_capture(chosen_method):
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    video_capture = cv2.VideoCapture(0)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    while(True):
        #capture frame-by-frame
        ret, frame = video_capture.read()
        #faces = face_detetion(frame, faceCascade)
        faces = detector(frame, 1)

        win.clear_overlay()
        win.set_image(frame)

        #Draw a rectangle around the faces
        for face in faces:
            if(chosen_method == 1):
                cutted_face = frame[dlib.rectangle.top(face):dlib.rectangle.top(face)+dlib.rectangle.height(face), dlib.rectangle.left(face):dlib.rectangle.left(face)+dlib.rectangle.width(face)]
                normalized_face = cv2.cvtColor(cutted_face, cv2.COLOR_BGR2GRAY)
                normalized_face = cv2.resize(normalized_face, (350, 350))
                prediction = fishface.predict(normalized_face)
                print emotions[prediction[0]]
            #print face, dlib.rectangle.bottom(face),dlib.rectangle.top(face),dlib.rectangle.left(face),dlib.rectangle.right(face),dlib.rectangle.height(face),dlib.rectangle.width(face)
            shape = predictor(frame, face)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #normalize the landmark vector
            normalized_shape = np.zeros(shape=(68,2))
            for i in range(0,shape.num_parts):
                x = (shape.part(i).x - dlib.rectangle.left(face)) / float(dlib.rectangle.width(face))
                if x<0:
                    x=0
                elif x>1:
                    x=1

                y = (shape.part(i).y - dlib.rectangle.top(face)) / float(dlib.rectangle.height(face))
                if(y<0):
                    y=0
                elif y>1:
                    y=1

                normalized_shape[i]=[x,y]

            win.add_overlay(shape)

            #print normalized_shape


        win.add_overlay(faces)


        #Display the resulting frame
        #cv2.imshow('Video', frame)

        #Press q to quit
        key = cv2.waitKey(5)
        if key == 27:
            print("pressed escape")
            break
    win.close()
    video_capture.release()
    #cv2.destroyAllWindows()

def check_faces(emotion):
    detector = dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    files = glob.glob("sorted_set/%s/*" %emotion)
    no_face = 0
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces)<1:
            no_face+=1
        for (x,y,w,h) in faces:
            gray = gray[y:y+h, x:x+w]
            try:
                out = cv2.resize(gray, (350, 350))
                cv2.imwrite("dataset/%s/%s.jpg" %(emotion, filenumber), out)
                filenumber +=1
            except:
                print "imagem que fudeu: ",emotion, " ", f
    print "No_face: %d" %no_face


def face_detetion(frame, faceCascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    return faces

def train_svm():
    return


#fisher faces implementation http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
def get_files(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[-int(len(files)*0.2):]
    return training, prediction

def make_sets(number_emotions):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    print number_emotions
    for emotion in range(0, number_emotions):
        print emotions[emotion]
        training, prediction = get_files(emotions[emotion])
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotion)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotion)
    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer(number_emotions):
    training_data, training_labels, prediction_data, prediction_labels = make_sets(number_emotions)

    print "Training fisher face classifier"
    print "Size of training set is: ",len(training_data)," images"
    fishface.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    count = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[count]:
            correct += 1
        else:
            incorrect += 1
        count +=1
    return ((100*correct)/(correct + incorrect)), len(training_data)
def train_fisher_self():
    video_capture = cv2.VideoCapture(0)
    training_data = []
    training_labels = []
    number_times = 20
    if len(sys.argv) > 2:
        number_times = int(sys.argv[2])
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()
    for(emotion) in emotions:
        print "ACT {} !!!".format(emotion)
        time.sleep(1)
        for i in range(0,number_times):
            print i
            #capture frame-by-frame
            ret, frame = video_capture.read()
            #faces = face_detetion(frame, faceCascade)
            faces = detector(frame, 1)

            win.clear_overlay()
            win.set_image(frame)

            for face in faces:
                try:
                    cutted_face = frame[dlib.rectangle.top(face):dlib.rectangle.top(face)+dlib.rectangle.height(face), dlib.rectangle.left(face):dlib.rectangle.left(face)+dlib.rectangle.width(face)]
                    normalized_face = cv2.cvtColor(cutted_face, cv2.COLOR_BGR2GRAY)
                    normalized_face = cv2.resize(normalized_face, (350, 350))
                    training_data.append(normalized_face)
                    training_labels.append(emotions.index(emotion))
                except:
                    pass
    video_capture.release()
    print "Training fisher face classifier"
    print "Size of training set is: ",len(training_data)," images"
    fishface.train(training_data, np.asarray(training_labels))

def incremental_fisher_faces_test():
    file_object = open("incremental_fisher_tests.txt", 'w')
    for i in range(2, len(emotions)):
        file_object.write("Test with {} diferent emotions".format(i))
        print "Training with ",i," emotions"
        metascore = []
        size_training_data = 0
        for j in range(0,10):
            correct, size_training_data = run_recognizer(i)
            print "got {} percent correct".format(correct)
            metascore.append(correct)
        print "Mean score: {} percent correct".format(np.mean(metascore))
        file_object.write("Mean score: {} percent correct with Training data with {} images and 10 tests".format(np.mean(metascore), size_training_data))



if __name__ == "__main__":
    main()
