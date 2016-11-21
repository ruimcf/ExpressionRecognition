import cv2
import sys


def main():
    if(len(sys.argv) < 2):
        print("Use xml classifier file as parameter")
        sys.exit(0)
    print("Press 'q' to quit")
    image_capture()

def image_capture():
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    while(True):
        #capture frame-by-frame
        ret, frame = video_capture.read()

        faces = face_detetion(frame, faceCascade)

        #Draw a rectangle around the faces
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #Display the resulting frame
        cv2.imshow('Video', frame)

        #Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

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
