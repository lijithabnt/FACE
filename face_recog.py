import cv2
import sys
import os
import datetime
import time

# Load the Haar cascader file
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert image to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
        flags=cv2.FONT_HERSHEY_SIMPLEX)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Add the timestamp at which face was detected
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')

    # Stop if esc+q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write the number of faces detected and timestamp of it
print("No of faces detected {} at {}".format(len(faces), st))
f = open("resultfile.txt", "w")
results="No of Faces detected  "+ str(len(faces))+" at "+ str(st)
f.write(results)
f.close()


# Once everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()