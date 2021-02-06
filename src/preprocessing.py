import cv2,os

#Specifying the path to the dataset directory
data_path='dataset'
#Getting the categories of the dataset i.e "with_mask" and "without_mask" and labeling each image as such
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

#creates a dictionary that is used to store all of the labels of each image
label_dict=dict(zip(categories,labels))

#setting the image size and creating 2 arrays that will store the data and the target values for the dataset
img_size=100
data=[]
target=[]

#Loop through each category
for category in categories:
    #Get the names of every image in the directory
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    #For every image in the list of images
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        #Convert the images to gray scale, resize them and add it and the lable into the list
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            #Coverting the image into gray scale
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)
       
        #Exception handling
        except Exception as e:
            print('Exception:',e)

import numpy as np

#Flatening the data and converting them to numpy files to be used in the training of the model
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)