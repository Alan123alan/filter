import cv2
import numpy as np 

#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

#read image
img = cv2.imread('people.jpg')

#resize image to fit in the screen
img_resized = cv2.resize(img, (960, 540)) 

#convert to gray scale to apply the face_cascade.detectMultiScale method
grayed_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#detect faces
#faces is a list of lists with coordinates and a width and height [x origin, y origin, width, height]
faces = face_cascade.detectMultiScale(grayed_img, 1.3, 5)

#for each face
for (x_origin, y_origin, width, height) in faces:
    #x_origin and y_origin
    face_origin_point = (x_origin,y_origin)
    #draw rectangle around face
    img = cv2.rectangle(img_resized, face_origin_point,(x_origin+width,y_origin+height),(255,0,0),2)
    #select face as region of interest 
    roi_g = grayed_img[y_origin:y_origin+height,x_origin:x_origin+height]
    roi_c = img[y_origin:y_origin+height,x_origin:x_origin+height]
    #within region of interest find eyes
    eyes = eye_cascade.detectMultiScale(roi_g)
    #for each eye
    for (ex,ey,ew,eh) in eyes:
        #draw retangle around eye
        cv2.rectangle(roi_c, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img_resized) #shows image
cv2.waitKey(0) #waits until a key is pressed to progress
cv2.destroyAllWindows() #closes windows