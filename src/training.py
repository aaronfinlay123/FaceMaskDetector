import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

# Load numpy arrays from preprocessing
# Data contains images
# Target refers to the 2 classifications: with mask and without mask
data = np.load('data.npy')
target = np.load('target.npy')

# Sequential model refers to a model that inputs and outputs sequences of data
model = Sequential()

# To follow is the first Convolutional layer and then ReLU layer
# And MaxPooling Layer

# First convolutional layer 200 kernels size 3 x 3
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:])) 

# Activation function of a node defines output of node when given inputs
# Rectified Linear Unit - Most commonly used activation function in CNN
model.add(Activation('relu'))

# Pooling operation which calculates max value in each patch of each feature map
model.add(MaxPooling2D(pool_size=(2,2)))


# What follows is the second Convolutional layer, with ReLU and MaxPooling layer
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening the data is to convert the current data to a 1D array for input 
model.add(Flatten())

# Dropout refers randomly selected neurons are ignored during training
# This is to prevent overfitting, here every 1 in 2 neurons will dropout
model.add(Dropout(0.5))

# Adding a dense layer 
model.add(Dense(50,activation='relu'))

# Final layer with 2 outputs, these outputs refer to with or without mask
model.add(Dense(2,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Sets the test size split to 0.2 to evaluate the data
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

#Runs 50 epochs on the network with a validation split of 0.2
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=False,mode='auto')
history=model.fit(train_data,train_target,epochs=50,callbacks=[checkpoint],validation_split=0.2)

#Creates a graph of the results of the epoch to visualise the training loss against the validation
from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#Creates a graph to visualise the training accuracy against the validation accuracy
plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()