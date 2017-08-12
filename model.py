import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers.core import Lambda, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Cropping2D

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images1 = []
            measurements1 = []
            for line in batch_samples:

                # read center, left and right image
                img_center = np.asarray(Image.open(line[0].split('/')[-3]+'/'+line[0].split('/')[-2]+'/'+line[0].split('/')[-1]))
                img_left = np.asarray(Image.open(line[1].split('/')[-3]+'/'+line[1].split('/')[-2]+'/'+line[1].split('/')[-1]))
                img_right = np.asarray(Image.open(line[2].split('/')[-3]+'/'+line[2].split('/')[-2]+'/'+line[2].split('/')[-1]))
                
                # flip center image
                image_flipped = np.fliplr(img_center)

                # read recorded steering
                steering_center = float(line[3])

                # correct steering for the left and right camera images
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

                # only add flipped image, if significantly steered -> to avoid zero-steering biasing
                if (steering_center >= 0.08 or steering_center <= -0.08):
                    images1.append(image_flipped);
                    measurements1.append(steering_flipped);

            X_train = np.array(images1)
            y_train = np.array(measurements1)
            yield sklearn.utils.shuffle(X_train, y_train)            
            
       
    
batch_size = 32
lines1 = []


with open('recordings/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile);
    for line in reader:
        lines1.append(line)


train_samples, validation_samples = train_test_split(lines1, test_size=0.15)

# obtain exact number of training samples, which the generator generates
my_samples_per_epoch = 0;
for line in train_samples:
        my_samples_per_epoch = my_samples_per_epoch + 3;  # three cameras 
        steering_center = float(line[3])
        if (steering_center >= 0.08 or steering_center <= -0.08): # flipped
            my_samples_per_epoch = my_samples_per_epoch + 1; 
          
        

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

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

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch=my_samples_per_epoch, 
                                     validation_data=validation_generator, 
                                     nb_val_samples=len(validation_samples), 
                                     nb_epoch=2)

model.save('model.h5')
