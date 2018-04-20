import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt
import sys
from keras.models import load_model

img_width, img_height = 32, 32
img = cv2.imread('try.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_width, img_height))
model=load_model('my_model.h5')
arr = numpy.array(img).reshape((3,img_width,img_height))
arr = numpy.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
print(prediction)
bestclass = ''
bestconf = -1
best = ['non-vehicle','vehicle','non-vehicle','non-vehicle','non-vehicle','non-vehicle','non-vehicle','non-vehicle','non-vehicle','vehicle']
for n in [0,1,2]:
	if (prediction[n] > bestconf):
		bestclass = n
		bestconf = prediction[n]
print (best[bestclass])