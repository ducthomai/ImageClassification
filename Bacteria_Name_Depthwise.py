from __future__ import print_function
import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, DepthwiseConv2D, Dropout, Flatten, SeparableConv2D
from keras import models
from keras.models import load_model
import os
from os import listdir
import cv2
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# import glob2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, DepthwiseConv2D, Dropout, Flatten, SeparableConv2D, GlobalAveragePooling2D, SeparableConv2D



raw_folder = "Identify"
def save_data(raw_foler =raw_folder):
    dest_size = (128, 128)
    print("Start to image processing...")

    pixels = []
    labels = []

    #Repeat across subfolder in raw folder
    for folder in listdir(raw_folder):
        if folder !='.DS_Store':
            print("Folder =", folder)
            #Repeat across the files in each folder
            for file in listdir(raw_folder + "/" + folder):
                if file != '.DS_Store':
                    print("File =", file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder +"/"  + folder +"/" + file),dsize=(128,128)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    #dump information to that file
    pickle.dump((pixels,labels), file)
    #close the file
    file.close()

    return

def load_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

save_data()



X, y = load_data()
# random.shuffle(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



#Design the model
#start point
my_mobilenet_model = models.Sequential()

#add first convolutional block
my_mobilenet_model.add(Conv2D(16,(3,3),activation = 'relu', strides= (2,2), padding = 'same', input_shape =(128,128,3)))
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
my_mobilenet_model.add(BatchNormalization())

#add second block
my_mobilenet_model.add(SeparableConv2D(filters=32, kernel_size= (3,3),strides=(1,1), padding='valid',activation='relu'))
my_mobilenet_model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
# my_mobilenet_model.add(Dropout(0.5))
# my_mobilenet_model.add(BatchNormalization())

#add third block
# my_mobilenet_model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
my_mobilenet_model.add(SeparableConv2D(filters=64, kernel_size= (3,3),strides=(1,1), padding='same',activation='relu', use_bias=False))
my_mobilenet_model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
#my_mobilenet_model.add(BatchNormalization())
#my_mobilenet_model.add(DepthwiseConv2D(64,(3,3),activation ='relu', padding = 'same'))
#my_mobilenet_model.add(Conv2D(64, (1,1), activation = 'relu', padding = 'same'))
#add fourth block
# my_mobilenet_model.add(DepthwiseConv2D(kernel_size= (1,1),strides=(1,1), padding='valid',activation='relu'))
# my_mobilenet_model.add(Conv2D(128, (1,1), activation = 'relu', padding = 'same'))
# my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
#add fifth layer
#my_mobilenet_model.add(DepthwiseConv2D(kernel_size= (1,1),strides=(2,2), padding='valid',activation='relu'))
#my_mobilenet_model.add(Conv2D(256, (1,1), activation = 'relu', padding = 'same'))
#my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
# add GlobalAvgPooling

# my_mobilenet_model.add(Dropout(0.5))
#my_mobilenet_model.add(GlobalMaxPooling2D())
my_mobilenet_model.add(Dense(16, activation = 'relu', kernel_regularizer='l2'))
#my_mobilenet_model.add(Dropout(0.5))
my_mobilenet_model.add(Flatten())
my_mobilenet_model.add(Dropout(0.5))
#my_mobilenet_model.save_weights(my_mobilenet_model, "Xray_chest", bool=True)

my_mobilenet_model.add(Dense(11,activation = 'softmax'))
my_mobilenet_model.summary()

#compile model
opt = keras.optimizers.Adam(learning_rate=0.0001)
my_mobilenet_model.compile(loss= 'categorical_crossentropy', metrics= ['accuracy'],optimizer=opt)
filepath="weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    rescale=1./255,
	width_shift_range=0.2,
    height_shift_range=0.2,
	horizontal_flip=True,
    brightness_range=[0.1,1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1./255)

history=my_mobilenet_model.fit_generator(aug.flow(X_train, y_train, batch_size=16),
                               epochs=100,# steps_per_epoch=len(X_train)//64,
                               validation_data=aug.flow(X_test,y_test,
                               batch_size=len(X_test)),
                               callbacks=callbacks_list)

my_mobilenet_model.save("mymodel.h5")

# Print the diagram of results
import matplotlib.pyplot as plt
# %matplotlib inline

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.ylim(0, 1)
plt.axis([0,100, 0, 1])
plt.plot(epochs, acc,  label='Training acc')
plt.plot(epochs, val_acc,  label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print('Results:\n')
print('The best validation accuracy is %.2f%%.' % (np.max(history.history["val_accuracy"])*100))
my_mobilenet_model.save('final_model.h5')

