import cv2
import sys
import dlib
from skimage import io


def main():
    if(len(sys.argv) < 2):
        print("Use xml classifier file as parameter")
        sys.exit(0)
    print("Press 'q' to quit")
    image_capture()

def image_capture():
    cascPath = sys.argv[1]
    preditor_path = sys.argv[2]
    #Face classifier
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    predictor_path = sys.argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(preditor_path)
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
            shape = predictor(frame, face)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            win.add_overlay(shape)

        win.add_overlay(faces)


        #Display the resulting frame
        cv2.imshow('Video', frame)

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


if __name__ == "__main__":
    main()
