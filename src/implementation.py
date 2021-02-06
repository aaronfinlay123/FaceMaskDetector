from keras.models import load_model
import cv2
import numpy as np

#Loading the model that has the highest accuracy and lowest loss
trainedmodel = load_model('model-050.model')

#The haar cascade classifier algorithm is being used to locate the face
faceclassifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Opens the default web camera for the device
webcamera=cv2.VideoCapture(0)

#Dictionaries that contain the labels and the colours for the bounding boxes
labels={0:'NO MASK',1:'MASK'}
colours={0:(0,0,255),1:(0,255,0)}

#keeps looping as long as the web camera is open
while(True):
    #reads in the frame from the webcam
    ret,frame=webcamera.read()
    #converts the image to gray scale to normalise the data
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceclassifier.detectMultiScale(grey,1.3,5)

    #for each frame in faces do this:
    for (x,y,w,h) in faces:
        #normalise and reshape the face
        face=grey[y:y+w,x:x+w]
        resized=cv2.resize(face,(100,100))
        normalised=resized/255.0
        reshaped=np.reshape(normalised,(1,100,100,1))
        #predict if the face is wearing a mask or not
        result=trainedmodel.predict(reshaped)

        #save the labels that were found in prediction
        label=np.argmax(result,axis=1)[0]

        #Draw the bounding box around the face and the result i.e 'MASK' and green box or 'NO MASK' and red box
        cv2.rectangle(frame,(x,y),(x+w,y+h),colours[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),colours[label],-1)
        cv2.putText(frame, labels[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    #display window containing the image
    cv2.imshow('Face mask detector by Kyle Hennessy, Jordan Marah, Aaron Finlay',frame)
    #wait for a key press
    key=cv2.waitKey(1)
    
    #if the key press is "escape" then we break out of the loop
    if(key==27):
        break

#Destroy the window after the loop has finished and release the webcam     
cv2.destroyAllWindows()
webcamera.release()