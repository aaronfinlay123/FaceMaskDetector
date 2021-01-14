#Model.py is the class responsible for the architecture of the CNN model at use
#Plan for structuring the CNN
#CONV LAYER -> CONV LAYER -> MAX POOL LAYER -> DROPOUT LAYER
#-> CONV LAYER -> CONV LAYER -> MAX POOL LAYER -> DROPOUT LAYER
#-> FULLY CONNECTED LAYER -> DROPOUT LAYER -> FULLY 
#CONNECTED LAYER -> SOFTMAX LAYER

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
##above calls an empty sequential model

##one layer at a time will be added, the f1st is a conv layer
#with filter size 3x3, stride size 1 (in both dimensions),
#and depth 32. The padding is the same and the activation is 'relu'
#(these two settings shall apply to all layers in the CNN)
model.add(Conv2D(32, (3, 3), activation=
'relu', padding='same', input_shape=(32,32,3)))

##What the above code does is add this layer to the empty sequential model,
##first no. 32 refers to depth, Next is the specification of activation in
#terms of which is 'relu' and padding which is 'same'.
# Stride remains 1 as a default setting, hence the non-specification of it,
#unless this setting is desired to be changed, there is no need to specify it.

#we must specify an inpuy size for the first (input) layer
#subsequent layers need not be specifie as they can infer the input
#size from the output size of the previous layer

model.add(Conv2D(32, (3, 3), activation='relu',
padding='same'))

#Next layer is the max pooling layer with pool size 2x2 & stride 2 (in both Dimensions).
#The default for max pooling layer stride is pool size, therefore, we do not
#need to specify the stride

model.add(MaxPooling2D(pool_size=(2, 2)))

#Finally, a dropout layer is needed with probability of 0.25 of
#dropout to prevent overfitting the model

model.add(Dropout(0.25))

#Next 4 layers, similar to previous except depth of conv layer = 64 rather than 32
model.add(Conv2D(64, (3, 3), activation='relu',
padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu',
padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#next the code for fully connected layer must be completed,
#the neurons are spatially arranged in a cube like structure,
#rather than a single row. To transform this structure into one row,
#it must first be flattened, therefore a Flatten layer shall be implemented
model.add(Flatten())
#a dense (FC) layer of 512 nerons w/ relu activation

model.add(Dense(512, activation='relu'))
#add another dropout of probability 0.5
model.add(Dropout(0.5))
#finally, a dense(FC) layer with 10 neurons and softmax activation:
model.add(Dense(10, activation='softmax'))

##summary of architecture
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#loss function used is categorical cross entropy loss,
#used widely for classification problems
#Optimizer used here is 'Adam',
#a type of stochastic gradient descent (modified) so that
#it is able to train better. Accuracy of model is another
#important metric to be tracked.

hist = model.fit(x_train, y_train_one_hot, 
                 batch_size=32, epochs=20,
                 validation_split=0.2)
#model is trained with batch size 32, 20 epochs.
#Using the setting validation_split=0.2 enables a quick and easy
#partition of the dataset, removing the need to manually split the train
#and validation sets at the beginning, 20% used to validate the model

#can visualize the model training & validation loss over
#the number of epochs using this code
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

model.evaluate(x_test, y_test_one_hot)[1]
model.save('my_cifar10_model.h5')
##above code saves trained model in HDF5 format

##to load the saved model
# from keras.models import load_model
# model = load_model('my_cifar10_model.h5')

my_image = plt.imread("C:/Users/aaron/HumanActivityRecognitioncat.jpg")