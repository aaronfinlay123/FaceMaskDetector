from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
#load photographs
pixels = imread('test2.jpg')
#load the pre trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
#perform face detection
boundryboxes = classifier.detectMultiScale(pixels, 1.1, 3)
#print bounding box for each detailed face
for box in boundryboxes:
    #extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
    #draw a rectangle over the pixels
    rectangle(pixels, (x, y), (x2,y2), (0,0,255), 1)
#show the image
imshow('face detection', pixels)
#keep the window open until we press a key
waitKey(0)
#close the window
destroyAllWindows()