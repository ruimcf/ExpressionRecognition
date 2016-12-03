import cv2
import sys
import dlib
from skimage import io
import numpy as np

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def main():
    if(len(sys.argv) < 2):
        print("Use xml classifier file as parameter")
        sys.exit(0)
    print("Press 'q' to quit")
    image_capture()

def image_capture():
    cascPath = "haarcascade_frontalface_default.xml"
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    #Face detector
    faceCascade = cv2.CascadeClassifier(cascPath)

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

            print normalized_shape


        win.add_overlay(faces)


        #Display the resulting frame
        #cv2.imshow('Video', frame)

        #Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("pressed q")
            break
    win.close()
    video_capture.release()
    #cv2.destroyAllWindows()

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

#http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
def train_fisher_faces():
    files = 1


if __name__ == "__main__":
    main()
