import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

zeros = 0
lines1 = []

with open('recordings/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile);
    for line in reader:
        lines1.append(line)
     

random.shuffle(lines1)

images1 = []
measurements1 = []
for line in lines1:

    img_center = np.asarray(Image.open(line[0].split('/')[-3]+'/'+line[0].split('/')[-2]+'/'+line[0].split('/')[-1]))
    img_left = np.asarray(Image.open(line[1].split('/')[-3]+'/'+line[1].split('/')[-2]+'/'+line[1].split('/')[-1]))
    img_right = np.asarray(Image.open(line[2].split('/')[-3]+'/'+line[2].split('/')[-2]+'/'+line[2].split('/')[-1]))
    image_flipped = np.fliplr(img_center)
    
    steering_center = float(line[3])

    steering_flipped = -steering_center 
    correction = 0.22 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    images1.append(img_center);
    measurements1.append(steering_center);

    images1.append(img_left);
    measurements1.append(steering_left);
    images1.append(img_right);
    measurements1.append(steering_right);    
    
    if (steering_center >= 0.08 or steering_center <= -0.08):
        images1.append(image_flipped);
        measurements1.append(steering_flipped);

X_train = np.array(images1)
Y_train = np.array(measurements1)

from keras.layers.core import Lambda, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.15, shuffle=True, nb_epoch=2)

model.save('model.h5')
