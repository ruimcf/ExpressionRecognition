import cv2
import time
import sys
import dlib
import glob
import random
from skimage import io
from sklearn.metrics import confusion_matrix
import numpy as np
import Image

emotions = ["neutral", "happy", "sadness", "surprise", "anger", "disgust", "fear"]
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
        params = dict( kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)
        self.model.train_auto(samples, responses, None, None, params, 3)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])

#main----------
def main():
    svm = 0
    chosen_method = 0
    if(len(sys.argv) > 1):

        #debug face extration in the dataset and dataset creation
        if(sys.argv[1] == "1"):
            print "debug face extration in the dataset and dataset creation"

            for emotion in emotions:
                check_faces(emotion);

        #Run fisher faces trainer on dataset
        elif(sys.argv[1] == "2"):
            print "Run fisher faces trainer on database"
            metascore = []
            conf_matrix = np.zeros(shape=(len(emotions)-1,len(emotions)-1))
            for i in range(0,10):
                correct , size_training, matrix = run_recognizer(len(emotions)-1)
                a = np.matrix(matrix)
                conf_matrix = conf_matrix + a
                print a, conf_matrix
                print "got {} percent correct".format(correct)
                metascore.append(correct)
            print "Mean score: {} percent correct".format(np.mean(metascore))
            '''saving the model obtained'''
            fishface.save('fishface_emotion_detect_model.xml')
            chosen_method = 1

        #load a fisherface model and run webcam
        elif(sys.argv[1] == "3"):
            print "load a fisherface model and run webcam"
            if(len(sys.argv) > 2):
                fishface.load(sys.argv[2])
            else:
                fishface.load('emotion_detection_model.xml')
            chosen_method = 1

        #train fisherface with webcam  and run webcam
        elif(sys.argv[1] == "4"):
            print "Train fisherface with webcam and run webcam"
            train_fisher_webcam()
            chosen_method = 1
            fishface.save('fishface_self_emotion_detect_model.xml')

        #test with incremental number of emotions
        elif(sys.argv[1] == "5"):
            print "test with incremental number of emotions"
            incremental_fisher_faces_test()
            chosen_method = -1

        #Run svm on dataset
        elif(sys.argv[1] == "12"):
            print "Run svm on database"
            metascore = []
            for i in range(0, 10):
                svm, correct = start_svm()
                print "got {} percent correct".format(correct)
                metascore.append(correct)
            print "Mean score: {} percent correct".format(np.mean(metascore))
            '''saving the model obtained'''
            print "Saving model"
            svm.save("svm_model_dataset.xml")
            chosen_method = 2

        #Load SVM model and run webcam
        elif(sys.argv[1] == "11"):
            print "Load svm model and run webcam"
            chosen_method = 2
            svm = SVM()
            if(len(sys.argv) > 2):
                svm.load(sys.argv[2])
            else:
                svm.load('svm_model_dataset.xml')

        #Train SVM with webcam
        elif(sys.argv[1] == "13"):
            print "Train SVM with webcam"
            svm = train_svm_webcam()
            chosen_method = 2
            svm.save("svm_model_webcam.xml")

        #Incremental svm tests
        elif(sys.argv[1] == "14"):
            print "Incremental SVM tests"
            incremental_svm_tests()
            chosen_method = -1


    if(chosen_method >= 0):
        #begin webcam capture
        print "Starting recognition"
        image_capture(chosen_method, svm)

def image_capture(chosen_method, svm):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    video_capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    emotion_images = [cv2.imread('emotion_images/%s.png' % emotion, -1) for emotion in emotions]
    while(True):
        #capture frame-by-frame
        ret, frame = video_capture.read()
        faces = detector(frame, 1)
        win.clear_overlay()
        win.set_image(frame)
        #for each face detected
        for face in faces:
            #fisher faces method
            if(chosen_method == 1):
                cutted_face = frame[dlib.rectangle.top(face):dlib.rectangle.top(face)+dlib.rectangle.height(face), dlib.rectangle.left(face):dlib.rectangle.left(face)+dlib.rectangle.width(face)]
                normalized_face = cv2.cvtColor(cutted_face, cv2.COLOR_BGR2GRAY)
                normalized_face = cv2.resize(normalized_face, (350, 350))
                prediction = fishface.predict(normalized_face)
                print emotions[prediction[0]]
            shape = predictor(frame, face)
            win.add_overlay(shape)
            #face landmark detection with svm method
            if(chosen_method == 2):
                prediction_array = np.empty(shape=(1,136),dtype=np.float32)
                shape_array = np.empty(shape=136)
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
                    shape_array[i*2]=np.float32(x)
                    shape_array[i*2+1]=np.float32(y)
                prediction_array[0] = shape_array.ravel()
                y_val = svm.predict(prediction_array)
                print emotions[int(y_val[0])]
        win.add_overlay(faces)

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
                print "imagem que partiu: ",emotion, " ", f
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
    fishface = cv2.createFisherFaceRecognizer()
    training_data, training_labels, prediction_data, prediction_labels = make_sets(number_emotions)
    print "Training fisher face classifier"
    print "Size of training set is: ",len(training_data)," images"
    fishface.train(training_data, np.asarray(training_labels))
    print "predicting classification set"
    count = 0
    correct = 0
    incorrect = 0
    predicted_labels = []
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        predicted_labels.append(pred)
        if pred == prediction_labels[count]:
            correct += 1
        else:
            incorrect += 1
        count +=1
    matrix = confusion_matrix(prediction_labels, predicted_labels)
    return ((100*correct)/(correct + incorrect)), len(training_data), matrix

def train_fisher_webcam():
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
                    print "PASSED"
                    pass
    video_capture.release()
    print "Training fisher face classifier"
    print "Size of training set is: ",len(training_data)," images"
    fishface.train(training_data, np.asarray(training_labels))

def incremental_fisher_faces_test():
    file_object = open("incremental_fisher_tests_2.txt", 'w')
    for i in range(2, len(emotions)+1):
        file_object.write("Test with {} diferent emotions\n".format(i))
        print "Training with ",i," emotions"
        metascore = []
        conf_matrix = np.zeros(shape=(i,i))
        size_training_data = 0
        for j in range(0,10):
            correct, size_training_data, matrix = run_recognizer(i)
            a = np.matrix(matrix)
            conf_matrix = conf_matrix + a
            print "got {} percent correct".format(correct)
            metascore.append(correct)
        print "Mean score: {} percent correct".format(np.mean(metascore))
        print conf_matrix
        np.savetxt("matrix_confusion_{}_emotions.txt".format(i), conf_matrix)
        file_object.write("Mean score: {} percent correct with Training data with {} images and 10 tests\n".format(np.mean(metascore), size_training_data))

#SVM-------------------------------------
def start_svm(number_emotions):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    svm = SVM()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    training_data, training_labels, prediction_data, prediction_labels = make_sets(number_emotions)
    training_data_ = []
    training_labels_ = []
    prediction_data_ = []
    prediction_labels_ = []
    num_train_ex = 0
    for frame in training_data:
        faces = detector(frame, 1)
        for face in faces:
            shape = predictor(frame, face)
            #normalize the landmark vector
            shape_list = []
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
                shape_list.append(x)
                shape_list.append(y)
            training_data_.append(shape_list)
            training_labels_.append(training_labels[num_train_ex])
        num_train_ex+=1
    training_array = np.empty(shape=(len(training_data_), 136), dtype = np.float32)
    training_labels_array = np.empty(shape=len(training_labels_), dtype = np.float32)
    for i in range(0,len(training_data_)):
        shape_array = np.empty(shape=(136), dtype=np.float32)
        for j in range(0,len(training_data_[i])):
            shape_array[j] = np.float32(training_data_[i][j])
        training_array[i] = shape_array
        training_labels_array[i] = training_labels_[i]
    num_train_ex = 0
    for frame in prediction_data:
        faces = detector(frame, 1)
        #Draw a rectangle around the faces
        if(len(faces) != 1):
            print "FUCK",len(faces)
        for face in faces:
            shape = predictor(frame, face)
            #normalize the landmark vector
            shape_list = []
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
                shape_list.append(x)
                shape_list.append(y)
            prediction_data_.append(shape_list)
            prediction_labels_.append(prediction_labels[num_train_ex])
        num_train_ex+=1
    prediction_array = np.empty(shape=(len(prediction_data_), 136), dtype = np.float32)
    prediction_labels_array = np.empty(shape=len(prediction_labels_), dtype = np.float32)
    for i in range(0,len(prediction_data_)):
        shape_array = np.empty(shape=(136), dtype=np.float32)
        for j in range(0,len(prediction_data_[i])):
            shape_array[j] = np.float32(prediction_data_[i][j])
        prediction_array[i] = shape_array.ravel()
        prediction_labels_array[i] = prediction_labels_[i]
    print "Training SVM"
    svm.train(training_array, training_labels_array)
    y_val = svm.predict(prediction_array)

    print "predicting classification set"
    count = 0
    correct = 0
    incorrect = 0
    for val in y_val:
        if val == prediction_labels_array[count]:
            correct += 1
        else:
            incorrect += 1
        count +=1
    matrix = confusion_matrix(prediction_labels_array, y_val)
    return svm, ((100*correct)/(correct + incorrect)), matrix

def incremental_svm_tests():
    file_object = open("incremental_svm_tests.txt", 'w')
    for i in range(2, len(emotions)+1):
        file_object.write("Test with {} diferent emotions\n".format(i))
        print "Training with ",i," emotions"
        metascore = []
        conf_matrix = np.zeros(shape=(i,i))
        size_training_data = 0
        for j in range(0,10):
            svm, correct, matrix = start_svm(i)
            a = np.matrix(matrix)
            conf_matrix = conf_matrix + a
            print "got {} percent correct".format(correct)
            metascore.append(correct)
        print "Mean score: {} percent correct".format(np.mean(metascore))
        print conf_matrix
        np.savetxt("svm_matrix_confusion_{}_emotions.txt".format(i), conf_matrix)
        file_object.write("Mean score: {} percent correct with Training data with {} images and 10 tests\n".format(np.mean(metascore), size_training_data))
def train_svm_webcam():
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    svm = SVM()
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
        for j in range(0,number_times):
            print j
            #capture frame-by-frame
            ret, frame = video_capture.read()
            faces = detector(frame, 1)
            win.clear_overlay()
            win.set_image(frame)
            for face in faces:
                shape = predictor(frame, face)
                #normalize the landmark vector
                shape_list = []
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
                    shape_list.append(x)
                    shape_list.append(y)
                training_data.append(shape_list)
                training_labels.append(emotions.index(emotion))
                win.add_overlay(shape)
            win.add_overlay(faces)
    training_array = np.empty(shape=(len(training_data), 136), dtype = np.float32)
    training_labels_array = np.empty(shape=len(training_labels), dtype = np.float32)
    for i in range(0,len(training_data)):
        shape_array = np.empty(shape=(136), dtype=np.float32)
        for j in range(0,len(training_data[i])):
            shape_array[j] = np.float32(training_data[i][j])
        training_array[i] = shape_array
        training_labels_array[i] = training_labels[i]
    video_capture.release()
    print "Training svm classifier"
    print "Size of training set is: ",len(training_data)," images"
    svm.train(training_array, training_labels_array)
    return svm

if __name__ == "__main__":
    main()
