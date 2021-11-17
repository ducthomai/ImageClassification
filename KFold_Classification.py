from __future__ import print_function


import numpy as np  # linear algebra
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import random
# Dependencies
import tensorflow as tf
# from tensorflow import keras
import keras
from keras.metrics import categorical_crossentropy, sparse_categorical_accuracy
from keras import models
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
# CNN
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import shutil
from PIL import ImageFile

warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datasetFolderName = 'DIBaS_Orig_aug_Adam31/'
MODEL_FILENAME = "model_cv.h5"
sourceFiles = []
#11 classes
# classLabels = ['Acinetobacter.baumanii', 'Actinomyces.israeli', 'Bacteroides.fragilis', 'Bifidobacterium.spp','Candida.albicans','Clostridium.perfringens',
#                'Enterococcus.faecalis','Enterococcus.faecium','Escherichia.coli', 'Fusobacterium','Lactobacillus.casei','Lactobacillus.crispatus',
#                'Lactobacillus.delbrueckii','Lactobacillus.gasseri','Lactobacillus.jehnsenii','Lactobacillus.johnsonii','Lactobacillus.paracasei']

#23 classes for ROD shaped
# classLabels = ['Acinetobacter.baumanii', 'Actinomyces.israeli', 'Bacteroides.fragilis', 'Bifidobacterium.spp','Clostridium.perfringens',
#                'Escherichia.coli', 'Fusobacterium','Lactobacillus.casei','Lactobacillus.crispatus',
#                'Lactobacillus.delbrueckii','Lactobacillus.gasseri','Lactobacillus.jehnsenii','Lactobacillus.johnsonii','Lactobacillus.paracasei','Lactobacillus.plantarum',
#                'Lactobacillus.reuteri','Lactobacillus.rhamnosus','Lactobacillus.salivarius','Listeria.monocytogenes','Porfyromonas.gingivalis',
#                'Propionibacterium.acnes','Pseudomonas.aeruginosa','Veionella']

#8 classes
# classLabels = ['Enterococcus.faecalis', 'Enterococcus.faecium','Micrococcus.spp','Neisseria.gonorrhoeae',
#                'Staphylococcus.aureus','Staphylococcus.epidermidis','Staphylococcus.saprophiticus','Streptococcus.agalactiae']

#33 classes
classLabels = ['Acinetobacter.baumanii', 'Actinomyces.israeli', 'Bacteroides.fragilis', 'Bifidobacterium.spp', 'Candida.albicans','Clostridium.perfringens',
                 'Enterococcus.faecalis','Enterococcus.faecium','Escherichia.coli', 'Fusobacterium','Lactobacillus.casei','Lactobacillus.crispatus',
                 'Lactobacillus.delbrueckii','Lactobacillus.gasseri','Lactobacillus.jehnsenii','Lactobacillus.johnsonii','Lactobacillus.paracasei','Lactobacillus.plantarum',
                 'Lactobacillus.reuteri','Lactobacillus.rhamnosus','Lactobacillus.salivarius','Listeria.monocytogenes','Micrococcus.spp','Neisseria.gonorrhoeae',
               'Porfyromonas.gingivalis' ,'Propionibacterium.acnes','Proteus','Pseudomonas.aeruginosa',
              # 'Staphylococcus.aureus',
               'Staphylococcus.epidermidis',
               #'Staphylococcus.saprophiticus',
   'Streptococcus.agalactiae',
               'Veionella'
                ]


def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)
    if (len(sourceFiles) != 0):
        transferFileNumbers = int(len(sourceFiles) * splitRate)
        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)


# First, check if test folder is empty or not, if not transfer all existing files to train
transferAllClassBetweenFolders('test', 'train', 1.0)
# Now, split some part of train data into the test folders.
transferAllClassBetweenFolders('train', 'test', 0.30)

X = []
Y = []


def prepareNameWithLabels(folderName):
    sourceFiles = os.listdir(datasetFolderName + '/train/' + folderName)
    for val in sourceFiles:
        X.append(val)
        if (folderName == classLabels[0]):
            Y.append(0)
        elif (folderName == classLabels[1]):
            Y.append(1)
        elif (folderName == classLabels[2]):
            Y.append(2)
        elif (folderName == classLabels[3]):
            Y.append(3)
        elif (folderName == classLabels[4]):
            Y.append(4)
        elif (folderName == classLabels[5]):
            Y.append(5)
        elif (folderName == classLabels[6]):
            Y.append(6)
        elif (folderName == classLabels[7]):
            Y.append(7)
        elif (folderName == classLabels[8]):
            Y.append(8)
        elif (folderName == classLabels[9]):
            Y.append(9)
        elif (folderName == classLabels[10]):
            Y.append(10)
        elif (folderName == classLabels[11]):
            Y.append(11)
        elif (folderName == classLabels[12]):
            Y.append(12)
        elif (folderName == classLabels[13]):
            Y.append(13)
        elif (folderName == classLabels[14]):
            Y.append(14)
        elif (folderName == classLabels[15]):
            Y.append(15)
        elif (folderName == classLabels[16]):
            Y.append(16)
        elif (folderName == classLabels[17]):
            Y.append(17)
        elif (folderName == classLabels[18]):
            Y.append(18)
        elif (folderName == classLabels[19]):
            Y.append(19)
        elif (folderName == classLabels[20]):
            Y.append(20)
        elif (folderName == classLabels[21]):
            Y.append(21)
        elif (folderName == classLabels[22]):
            Y.append(22)
        elif (folderName == classLabels[23]):
            Y.append(23)
        elif (folderName == classLabels[24]):
            Y.append(24)
        elif (folderName == classLabels[25]):
            Y.append(25)
        elif (folderName == classLabels[26]):
            Y.append(26)
        elif (folderName == classLabels[27]):
            Y.append(27)
        elif (folderName == classLabels[28]):
            Y.append(28)
        elif (folderName == classLabels[29]):
            Y.append(29)
        # elif (folderName == classLabels[30]):
        # #     Y.append(30)
        # elif (folderName == classLabels[31]):
        #     Y.append(31)
        else:
            Y.append(30)


# Organize file names and class labels in X and Y variables
prepareNameWithLabels(classLabels[0])
prepareNameWithLabels(classLabels[1])
prepareNameWithLabels(classLabels[2])
prepareNameWithLabels(classLabels[3])
prepareNameWithLabels(classLabels[4])
prepareNameWithLabels(classLabels[5])
prepareNameWithLabels(classLabels[6])
prepareNameWithLabels(classLabels[7])
prepareNameWithLabels(classLabels[8])
prepareNameWithLabels(classLabels[9])
prepareNameWithLabels(classLabels[10])
prepareNameWithLabels(classLabels[11])
prepareNameWithLabels(classLabels[12])
prepareNameWithLabels(classLabels[13])
prepareNameWithLabels(classLabels[14])
prepareNameWithLabels(classLabels[15])
prepareNameWithLabels(classLabels[16])
prepareNameWithLabels(classLabels[17])
prepareNameWithLabels(classLabels[18])
prepareNameWithLabels(classLabels[19])
prepareNameWithLabels(classLabels[20])
prepareNameWithLabels(classLabels[21])
prepareNameWithLabels(classLabels[22])
prepareNameWithLabels(classLabels[23])
prepareNameWithLabels(classLabels[24])
prepareNameWithLabels(classLabels[25])
prepareNameWithLabels(classLabels[26])
prepareNameWithLabels(classLabels[27])
prepareNameWithLabels(classLabels[28])
prepareNameWithLabels(classLabels[29])
prepareNameWithLabels(classLabels[30])
# prepareNameWithLabels(classLabels[31])
# prepareNameWithLabels(classLabels[32])

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape)
print(Y.shape)
# learning rate
batch_size = 16
epoch =50
activationFunction = 'relu'


# def getModel():
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), padding='same',strides= (2,2), activation=activationFunction, input_shape=(img_rows, img_cols, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Conv2D(64, (3, 3), activation=activationFunction))
#     model.add(BatchNormalization())
#
#     model.add(DepthwiseConv2D(kernel_size =(3,3),padding='same', strides=(1,1), activation=activationFunction))
#     # model.add(Conv2D(32, (3, 3), padding='same', activation=activationFunction))
#     # model.add(Conv2D(32, (3, 3), activation=activationFunction))
#     model.add(MaxPooling2D(pool_size=(2, 2), padding= 'same'))
#     # model.add(Dropout(0.25))
#
#     model.add(DepthwiseConv2D(kernel_size = (3, 3), strides= (1,1), padding='same', activation=activationFunction))
#     # model.add(Conv2D(16, (3, 3), activation=activationFunction))
#     model.add(MaxPooling2D(pool_size=(2, 2), padding= 'same'))
#     # model.add(Dropout(0.25))
#
#     # model.add(Flatten())
#     model.add(Dense(1000, activation=activationFunction))  # we can drop
#     model.add(Flatten())
#     model.add(Dropout(0.5))  # this layers
#     # model.add(Dense(32, activation=activationFunction))
#     # model.add(Dropout(0.1))
#     # model.add(Dense(16, activation=activationFunction))
#     # model.add(Dropout(0.1))
#     model.add(Dense(33, activation='softmax'))
#     model.summary()
#     return model

#Design the model
#start point
my_mobilenet_model = models.Sequential()

#add first convolutional block
my_mobilenet_model.add(Conv2D(64,(3,3),activation = 'relu', strides= (2,2), padding = 'same', input_shape =(224,224,3)))
my_mobilenet_model.add(BatchNormalization())
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))

#add second block

my_mobilenet_model.add(SeparableConv2D(64,kernel_size= (3,3),strides=(1,1), padding='same',activation='relu'))

# my_mobilenet_model.add(Conv2D(64,(1,1), activation = 'relu', padding = 'same')) #add a Conv2D with filter 3x3 and neuron = 16 will increase the accuracy
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
# my_mobilenet_model.add(Dropout(0.5))
# my_mobilenet_model.add(BatchNormalization())
# my_mobilenet_model.add(DepthwiseConv2D(kernel_size= (3,3),strides=(1,1), padding='same',activation='relu'))
# my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
#add third block
# my_mobilenet_model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
# my_mobilenet_model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
my_mobilenet_model.add(SeparableConv2D(64,kernel_size= (3,3),strides=(1,1), padding='same',activation='relu'))
# my_mobilenet_model.add(AveragePooling2D(pool_size=(7,7), padding='same', data_format=None))
# my_mobilenet_model.add(Conv2D(64, (1,1), activation = 'relu', padding = 'same'))
#my_mobilenet_model.add(BatchNormalization())
#my_mobilenet_model.add(DepthwiseConv2D(64,(3,3),activation ='relu', padding = 'same'))
#my_mobilenet_model.add(Conv2D(64, (1,1), activation = 'relu', padding = 'same'))
my_mobilenet_model.add(MaxPooling2D((2,2),padding = 'same'))
# my_mobilenet_model.add(MaxPooling2D((8,8),padding = 'same'))
my_mobilenet_model.add(Flatten())
my_mobilenet_model.add(Dense(256, activation = 'relu', kernel_regularizer='l2'))
# my_mobilenet_model.add(Dropout(0.5))
# my_mobilenet_model.add(Dense(100, activation = 'relu', kernel_regularizer='l2'))
my_mobilenet_model.add(Dropout(0.25))
#my_mobilenet_model.save_weights(my_mobilenet_model, "Xray_chest", bool=True)

my_mobilenet_model.add(Dense(31,activation = 'softmax'))
my_mobilenet_model.summary()

#compile model
opt = keras.optimizers.adam(learning_rate=0.0001) # CHANGE RMSPROP, SDG, ADAMAX TO VERIFY FOLLOWING THE SCENARIOS
my_mobilenet_model.compile(loss= 'categorical_crossentropy', metrics= ['accuracy'],optimizer=opt)
filepath="Weight/weightsPart1-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import matplotlib as plt
import seaborn as sns
def my_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    f1Score = f1_score(y_true, y_pred, average='weighted')
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # plt.figure(figsize=(35,35))
    # sns.heatmap(cm,annot=True,fmt=".3g", cmap='Blues')
    return accuracy, precision, f1Score


# input image dimensions
img_rows, img_cols = 224, 224

train_path = datasetFolderName + '/train/'
validation_path = datasetFolderName + '/validation/'
test_path = datasetFolderName + '/test/'
# model = my_mobilenet_model()

# ===============Stratified K-Fold======================
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X, Y)
foldNum = 0
for train_index, val_index in skf.split(X, Y):
    # First cut all images from validation to train (if any exists)
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    foldNum += 1
    print("Results for fold", foldNum)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    print(X_train.shape)
    print(X_val.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    # Move validation images of this fold from train folder to the validation folder
    for eachIndex in range(len(X_val)):
        classLabel = ''
        if (Y_val[eachIndex] == 0):
            classLabel = classLabels[0]
        elif (Y_val[eachIndex] == 1):
            classLabel = classLabels[1]
        elif (Y_val[eachIndex] == 2):
            classLabel = classLabels[2]
        elif (Y_val[eachIndex] == 3):
            classLabel = classLabels[3]
        elif (Y_val[eachIndex] == 4):
            classLabel = classLabels[4]
        elif (Y_val[eachIndex] == 5):
            classLabel = classLabels[5]
        elif (Y_val[eachIndex] == 6):
            classLabel = classLabels[6]
        elif (Y_val[eachIndex] == 7):
            classLabel = classLabels[7]
        elif (Y_val[eachIndex] == 8):
            classLabel = classLabels[8]
        elif (Y_val[eachIndex] == 9):
            classLabel = classLabels[9]
        elif (Y_val[eachIndex] == 10):
            classLabel = classLabels[10]
        elif (Y_val[eachIndex] == 11):
            classLabel = classLabels[11]
        elif (Y_val[eachIndex] == 12):
            classLabel = classLabels[12]
        elif (Y_val[eachIndex] == 13):
            classLabel = classLabels[13]
        elif (Y_val[eachIndex] == 14):
            classLabel = classLabels[14]
        elif (Y_val[eachIndex] == 15):
            classLabel = classLabels[15]
        elif (Y_val[eachIndex] == 16):
            classLabel = classLabels[16]
        elif (Y_val[eachIndex] == 17):
            classLabel = classLabels[17]
        elif (Y_val[eachIndex] == 18):
            classLabel = classLabels[18]
        elif (Y_val[eachIndex] == 19):
            classLabel = classLabels[19]
        elif (Y_val[eachIndex] == 20):
            classLabel = classLabels[20]
        elif (Y_val[eachIndex] == 21):
            classLabel = classLabels[21]
        elif (Y_val[eachIndex] == 22):
            classLabel = classLabels[22]
        elif (Y_val[eachIndex] == 23):
            classLabel = classLabels[23]
        elif (Y_val[eachIndex] == 24):
            classLabel = classLabels[24]
        elif (Y_val[eachIndex] == 25):
            classLabel = classLabels[25]
        elif (Y_val[eachIndex] == 26):
            classLabel = classLabels[26]
        elif (Y_val[eachIndex] == 27):
            classLabel = classLabels[27]
        elif (Y_val[eachIndex] == 28):
            classLabel = classLabels[28]
        elif (Y_val[eachIndex] == 29):
            classLabel = classLabels[29]
        # elif (Y_val[eachIndex] == 30):
        #     classLabel = classLabels[30]
        # elif (Y_val[eachIndex] == 31):
        #     classLabel = classLabels[31]
        else:
            classLabel = classLabels[30]
            # Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName + '/train/' + classLabel + '/' + X_val[eachIndex],
                    datasetFolderName + '/validation/' + classLabel + '/' + X_val[eachIndex])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                        zoom_range=0.20,
                                       shear_range= 0.20,
                                        fill_mode="nearest")
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Start ImageClassification Model
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        subset='training')

    validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                                    target_size=(img_rows, img_cols),
                                                                    batch_size=batch_size,
                                                                    class_mode='categorical',  # only data, no labels
                                                                    shuffle=False)
    # aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    #                          rescale=1. / 255,
    #                          width_shift_range=0.2,
    #                          height_shift_range=0.2,
    #                          horizontal_flip=True,
    #                          brightness_range=[0.1, 1.5], fill_mode="nearest")
    # # fit model
    history = my_mobilenet_model.fit_generator(train_generator,
                                               validation_data=validation_generator,
                                               epochs=epoch,
                                               callbacks= callbacks_list)
    # Print the diagram of results


import matplotlib as plt
predictions = my_mobilenet_model.predict_generator(validation_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
    # evaluate validation performance
print("***Performance on Validation data***")
valAcc, valPrec, valFScore = my_metrics(true_classes, yPredictions)
# plt.figure()
#     # plt.ylim(0, 1)
#     # plt.axis([0,200, 0, 1])
# plt.plot(valAcc, valPrec, valFScore,label='Training acc')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.plot(, val_acc, label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# =============TESTING=============
print("==============TEST RESULTS============")
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
predictions = my_mobilenet_model.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
testAcc, testPrec, testFScore = my_metrics(true_classes, yPredictions)
my_mobilenet_model.save(MODEL_FILENAME)


import matplotlib.pyplot as plt

    # %matplotlib inline

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
    # plt.ylim(0, 1)
    # plt.axis([0,200, 0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()

print('Results:\n')
print('The best validation accuracy is %.2f%%.' % (np.max(history.history["val_accuracy"]) * 100))