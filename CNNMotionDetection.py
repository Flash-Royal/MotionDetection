import numpy as np
import pandas as pd
import cv2, datetime, os
import os
from tqdm import tqdm
import matplotlib as plt
import sklearn
from sklearn.metrics import roc_curve

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Input, Concatenate, Layer
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.python.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

df = pd.read_csv('train\motions.txt', sep = ', ')
testdf = df.sample(frac = 0.15)
traindf = df[~df.prevImages.isin(testdf.prevImages)]

trainPrevImages = []
trainCurImages = []
trainLabels = []

testPrevImages = []
testCurImages = []
testLabels = []



for index, row in tqdm(traindf.iterrows(), total=traindf.shape[0]):
    imgPrev = cv2.imread(f"train/prevImages/{row['prevImages']}")
    imgCur = cv2.imread(f"train/curImages/{row['curImages']}")

    scale_percent = 40 # percent of original size
    # (56, 100, 3) - 20% (281, 500, 3) - 100%
    width = int(imgPrev.shape[1] * scale_percent / 100)
    height = int(imgPrev.shape[0] * scale_percent / 100)
    dim = (width, height)

    imgPrev = cv2.resize(imgPrev, dim, interpolation = cv2.INTER_AREA)
    imgCur = cv2.resize(imgCur, dim, interpolation = cv2.INTER_AREA)

    trainPrevImages.append(imgPrev / 255.0)
    trainCurImages.append(imgCur / 255.0)
    trainLabels.append(row['isMotion'])

for index, row in tqdm(testdf.iterrows(), total=testdf.shape[0]):
    imgPrev = cv2.imread(f"train/prevImages/{row['prevImages']}")
    imgCur = cv2.imread(f"train/curImages/{row['curImages']}")

    scale_percent = 40 # percent of original size
    width = int(imgPrev.shape[1] * scale_percent / 100)
    height = int(imgPrev.shape[0] * scale_percent / 100)
    dim = (width, height)

    imgPrev = cv2.resize(imgPrev, dim, interpolation = cv2.INTER_AREA)
    imgCur = cv2.resize(imgCur, dim, interpolation = cv2.INTER_AREA)

    testPrevImages.append(imgPrev / 255.0)
    testCurImages.append(imgCur / 255.0)
    testLabels.append(row['isMotion'])

print(trainLabels[0])
trainLabels = to_categorical(np.array(trainLabels), 2)
print(trainLabels[0])
testLabels = to_categorical(np.array(testLabels), 2)

# inputPrev = Input((len(trainPrevImages), 56, 100))
# inputCur = Input((len(trainCurImages), 56, 100))

inputPrev = Input((112, 200, 3))
inputCur = Input((112, 200, 3))

# inputPrev = Input((len(trainPrevImages),))
# inputCur = Input((len(trainCurImages),))
act = 'relu'
neuron = 128

prev = Conv2D(neuron, (3, 3), padding='same',
                 input_shape=(56, 100, 3), activation=act)(inputPrev)
cur = Conv2D(neuron, (3, 3), padding='same',
                 input_shape=(56, 100, 3), activation=act)(inputCur)

prev = MaxPooling2D(pool_size=(3, 3))(prev)
cur = MaxPooling2D(pool_size=(3, 3))(cur)

prev = Conv2D(neuron * 2, (3, 3), padding='same', activation=act)(prev)
cur = Conv2D(neuron * 2, (3, 3), padding='same', activation=act)(cur)

prev = MaxPooling2D(pool_size=(2, 2))(prev)
cur = MaxPooling2D(pool_size=(2, 2))(cur)

prev = Flatten()(prev)
cur = Flatten()(cur)

# prev = Dense(neuron * 4, activation = act)(prev)
# cur = Dense(neuron * 4, activation = act)(cur)

out = Concatenate()([prev,cur])

# out = Flatten()(out)

out = Dense(neuron * 2, activation = act)(out)

out = Dropout(0.25)(out)

out = Dense(neuron // 2, activation = act)(out)

out = Dropout(0.25)(out)

out = Dense(neuron // 8, activation = act)(out)

out = Dense(2, activation = 'softmax')(out)

model = Model([inputPrev,inputCur], out)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='RMSPROP',
              metrics=['accuracy'])

logdir = os.path.join("keks", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = TensorBoard(logdir, write_graph=False, histogram_freq=1, write_images = True)

model.fit([np.array(trainPrevImages), np.array(trainCurImages)], trainLabels, validation_split = 0.2, batch_size = 16, epochs = 10)

model.save("any4.h5")

results = model.evaluate([np.array(testPrevImages), np.array(testCurImages)], np.array(testLabels), batch_size=16)
print(f"Acc:{results[1]}, Loss:{results[0]}")
testPredict = model.predict([np.array(testPrevImages), np.array(testCurImages)], batch_size=16)
print(testPredict[0])

plot_roc("Test Baseline", testLabels, testPredict)
