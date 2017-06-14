import os
import cv2
import face_recognition as fr
import numpy as np



# 0 : default webcam (for laptop's camera or only one camera is connected)
# 1 : external webcam
cam = cv2.VideoCapture(0)


turn = True;

while cam.isOpened():
    ret, frame = cam.read()
    face_locations=fr.face_locations(frame)

    for (t,r,b,l) in face_locations:
	cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 2)
	#cv2.rectangle(frame, (l, b - 35), (r, b), (0, 0, 255), cv2.FILLED)
	#cv2.putText(frame, name, (l+6,b-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)

    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	break
	
cam.release()
cv2.destroyAllWindows

