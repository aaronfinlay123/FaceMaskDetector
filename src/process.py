##  Aaron Finlay
## HAR
## 4th Year Project
## 01/01/21
## **** Exploring AND Processing Data **** ##
## Primary Objective - download the datatset & visualize the images
## Change the label to one-hot encoding
## Scale the image pixel values to take between 0 and 1
## CIFAR-10 Dataset used for base implementation
##Img to be recognized - 32*32 pixels
##Labels - 10 possible (airplane, automobile, bird, cat, deer
## frog, horse, ship, and truck)
## Dataset size - 60,000 images, 50k training, 10k testing

#Step 1 - get dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

##The data needed is now stored in their own respetive arrays
## e.g. (x_train, y_train, x_test and y_test)
##Next Step Explor the data, figure out shape of input feature array
print ('x_train shape:', x_train.shape)
## result of print statement:
    ## 50k img, 32 pixels in h x w and 3 pixels in depth (e.g. RBG)
##Shape of Label array
print('y_train shape:', y_train.shape)
##output = one number (corresponding to label) for each out of 
##the 50,000 images
##try see example of image and label to solidify things. 
## Example 1
print(x_train[0])
## above will display matrices of 3x3 which aren't too useful however
##using matplotlib it should help better the view of the image
import matplotlib.pyplot as plt
%matplotlib inline

img = plt.imshow(x_train[0])

print('The label is:', y_train[0])

img = plt.imshow(x_train[1])
print('The label is:', y_train[1])

#Exploration of data complete now processing to take part
#will attempt to divide the class labels appropiately rather than no.
#goal is to get probability of 10 different classes
# therefore we will need 10 output neurons in our network
#since we have 10 output neurons, labels must match
#WE DO THIS BY:::
    #convert label into set of 10 nums where each num 
    #represents if the img belongs to that class or not
    #thus, if an image belongs to first class, the first num
    #of this set will be a 1 and all other numbers in the set 0
#This is known as one-hot encoding

import keras
y_train_one_hot = keras.utils.to_categorical(y_train,10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)
## first line means the initial array is taken with just number,
## y_train, and convert it to the one_hot_encodings, y_train_one_hot. 
##The number 10 is required as a parameter as you need to tell the
##function how many classes there are (10).

##If we want to see how the label for our second image (the truck, label:9)
## looks like in the one-hotsetting:
print('The one hot label is:', y_train_one_hot[1])

#Now that the labels are processed (y), we want to process our image (x).
#Common solution is to let the values be between 0-1, which will aid
#in the training of the neural network. As the pixel values already take 
#the values between 0 and 255, we simply need to divide by 255.
x_train = x_train.astype('float32')
x_test = x_test.astype('Float32')
x_train = x_train / 255
x_test = x_test / 255
## ^^ converts the type to float32, which is in turn a datatype
## that can store values with decimal points. Then each cell is divided by 255.
## can view the array values of the first train image by:
x_train[0]

##End of program
## Summary of objectives:
    ##Download dataset and visualize images
    ##Changed label to one-hot encodings
    ##Scale the image pixel values to take between 0-1
    



