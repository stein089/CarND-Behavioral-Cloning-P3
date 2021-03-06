# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Writeup / README

####  1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! 
And here is a link to my training script and model architecture [model.py](https://github.com/stein089/CarND-Behavioral-Cloning-P3/blob/master/model.py)

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/stein089/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/stein089/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/stein089/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/stein089/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA architecture and consits of five convolution layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 89-93) 

The model includes ELU layers to introduce nonlinearity.

The data is normalized in the model using a Keras lambda layer (code line 87). 

fter the convolution layers, the network is flattened and followed by five fully connected layers with sizes 500, 100, 50, 10 and 1 (code line 95-100). 

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting (model.py lines 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road, smooth driving through curves.
Difficult sections of the track were recorded multiple times (e.g., bridge, curve after bridge).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a already prooven network architecture as base and improve it gradually by fine-tuning the parameters and improve the training data quality.

My first step was to use a convolution neural network model similar to the LeNet architecture.  
I thought this model might be appropriate because it already worked for the traffic sign classification. 
But since it didn't perform as good as expected, I replaced the base model by the NVIDIA architecture. 
This architecture fits better to the given problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
Since I just used the center image at first, the MSE was low on training and validation data (<0.01). 
But the vehicle didn't perform well on the track, since it didn't know how to recover when it went of the trained path.

In order to conquer that problem, I augmented the training data with the side-camera images and modified steering angles. 
That increased the MSE for the model, but helped the car to peform better on the test track. 

To combat the overfitting, I added a dropout layer to the model and reduced to training epochs to two (since the validation error increased afterwards).

The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

The following snipped shows the output of `print(history_object.history.keys())`

```Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_9 (Lambda)                (None, 160, 320, 3)   0           lambda_input_9[0][0]             
____________________________________________________________________________________________________
cropping2d_9 (Cropping2D)        (None, 65, 320, 3)    0           lambda_9[0][0]                   
____________________________________________________________________________________________________
convolution2d_38 (Convolution2D) (None, 31, 158, 24)   1824        cropping2d_9[0][0]               
____________________________________________________________________________________________________
convolution2d_39 (Convolution2D) (None, 14, 77, 36)    21636       convolution2d_38[0][0]           
____________________________________________________________________________________________________
convolution2d_40 (Convolution2D) (None, 5, 37, 48)     43248       convolution2d_39[0][0]           
____________________________________________________________________________________________________
convolution2d_41 (Convolution2D) (None, 3, 35, 64)     27712       convolution2d_40[0][0]           
____________________________________________________________________________________________________
convolution2d_42 (Convolution2D) (None, 1, 33, 64)     36928       convolution2d_41[0][0]           
____________________________________________________________________________________________________
flatten_9 (Flatten)              (None, 2112)          0           convolution2d_42[0][0]           
____________________________________________________________________________________________________
dense_41 (Dense)                 (None, 500)           1056500     flatten_9[0][0]                  
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 500)           0           dense_41[0][0]                   
____________________________________________________________________________________________________
dense_42 (Dense)                 (None, 100)           50100       dropout_9[0][0]                  
____________________________________________________________________________________________________
dense_43 (Dense)                 (None, 50)            5050        dense_42[0][0]                   
____________________________________________________________________________________________________
dense_44 (Dense)                 (None, 10)            510         dense_43[0][0]                   
____________________________________________________________________________________________________
dense_45 (Dense)                 (None, 1)             11          dense_44[0][0]                   
====================================================================================================
Total params: 1,243,519
Trainable params: 1,243,519
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

<img src="./writeup_media/center.png" width="320" />


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the road on its own.
These images show what a recovery looks like starting from the right border of the road :

<img src="./writeup_media/recovery1.png" width="320" />
<img src="./writeup_media/recovery2.png" width="320" />
<img src="./writeup_media/recovery3.png" width="320" />

To augment the data sat, I also flipped images and angles thinking that this would add more training data and robustness to the model. But in order to not bias the model with zero-angle steering, I only added the flipped image for steering angles greater than 0.08 (see model.py line 50).

For example, here is an image that has then been flipped:

<img src="./writeup_media/flip1.png" width="320" />
<img src="./writeup_media/flip2.png" width="320" />

Additionally I also used the side-camera images with modified steering angle to augment the training data. 

The following plot shows the distribution of all steering angles within the training dataset. 
It can be seen that the model is not overly biased to a certain steering range. 
This is important to achieve a high performance of the network. 

<img src="./writeup_media/steering_histo.png" width="320" />


After the collection process, I had 15992 sample images. 
And I used 13593 samples to train, and 2399 samples to validate the model.


I then preprocessed this data by normalizing the image data to values between -1 and 1, and cropped the images to only the relevant data (road in front of the car).

I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 2 as evidenced by training for many more epochs and check where the val_loss increases. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

The following snipped the output of my training/validation process:
```Train on 13593 samples, validate on 2399 samples
Epoch 1/2
13593/13593 [==============================] - 30s - loss: 0.0271 - val_loss: 0.0180
Epoch 2/2
13593/13593 [==============================] - 29s - loss: 0.0169 - val_loss: 0.0167
dict_keys(['loss', 'val_loss'])
```

This graph shows the training and validation loss graphically for training the two epochs:

<img src="./writeup_media/loss.png" width="320" />
