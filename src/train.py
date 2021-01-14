#Progressively Loading Images
#It is possible to write code to manually load image data and return data ready for modeling.
#This would include walking the directory structure for a dataset, loading image data, 
#and returning the input (pixel arrays) and output (class integer).
#Thankfully, we don’t need to write this code. Instead, we can use the ImageDataGenerator class provided by Keras.
#The main benefit of using this class to load the data is that images are loaded for a single dataset in batches, 
#meaning that it can be used for loading both small datasets as well as very large image datasets with thousands or millions of image
#create a data generator
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
#Next, an iterator is required to progressively load images for a single dataset.
#This requires calling the flow_from_directory() function and specifying the dataset directory, such as the train, test, or validation directory.
#The function also allows you to configure more details related to the loading of images. Of note is the ‘target_size‘ argument 
#that allows you to load all images to a specific size, which is often required when modeling. 
#The function defaults to square images with the size (256, 256).
#The function also allows you to specify the type of classification task via the ‘class_mode‘ argument, 
#specifically whether it is ‘binary‘ or a multi-class classification ‘categorical‘.
#The default ‘batch_size‘ is 32, which means that 32 randomly selected images from across the classes in
#the dataset will be returned in each batch when training. Larger or smaller batches may be desired. 
#You may also want to return batches in a deterministic order when evaluating a model, which you can do by setting ‘shuffle‘ to ‘False.’
#We can use the same ImageDataGenerator to prepare separate iterators for separate dataset directories. 
#This is useful if we would like the same pixel scaling applied to multiple datasets (e.g. trian, test, etc.).
#load and iterate training dataset
train_iterator = datagen.flow_from_directory('~/Dataset/With_Face_Mask', class_mode='binary', batch_size=64)
#load and iterate validation dataset
validation_iterator = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
#load and iterate test dataset
test_iterator = datagen.flow_from_directory('~/Dataset/No_Face_Mask', class_mode='binary', batch_size=64)
#Once the iterators have been prepared, we can use them when fitting and evaluating a deep learning model.
#For example, fitting a model with a data generator can be achieved by calling the fit_generator() function 
#on the model and passing the training iterator (train_it). The validation iterator (val_it) can be specified
# when calling this function via the ‘validation_data‘ argument.
# building a linear stack of layers with the sequential model
model = sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit_generator(train_iterator, steps_per_epoch=16, validation_data=validation_iterator, validation_steps=8)
#Once the model is fit, it can be evaluated on a test dataset using the evaluate_generator() function 
#and passing in the test iterator (test_it). The ‘steps‘ argument defines the number of batches of samples 
#to step through when evaluating the model before stopping.

#evaluation of model
loss = model.evaluate_generator(test_iterator, steps=24)

#creation of iterator to fit model for making predictions
# make a prediction
prediction = model.predict_generator(predict_iterator, steps=24)