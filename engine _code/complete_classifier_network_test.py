import keras
from keras import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Lambda
from keras.layers import Conv1D, MaxPooling1D, Concatenate,UpSampling2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
import cv2


#read the output bounding box coordinates from faster rcnn 
dataset = pd.read_csv('frcnnoutput.csv')
data = dataset.values
print(data.shape)
print(data)
print(dataset.head())


dirName = 'imagestest/';#images folder path




#generating actualboxes
'''
for img in numinum:
	
	imstring = img.replace(".jpg","")
	index = int(imstring)
	#indexing.append(index)
	for bx in boxlabels:
		if(bx[0]==index):
			actualboxes.append(bx)

'''


cropped_list =[]#list of boxes cropped 

output = []

#generating cropped_list for classifier input
for bb in range(0,data.shape[0]):
	idx = data[bb][0]
	#print(idx)
	filepath = dirName + str(idx)
	imd = image.load_img(filepath)
	imarr = np.array(imd)
	
	a = int(data[bb][2])
	b = int(data[bb][4])
	c = int(data[bb][1])
	d = int(data[bb][3])


	rl = int(a*223/639)
	ru = int(b*223/639) + 1
	cl = int(c*223/479)
	cu = int(d*223/479) + 1
	
	
	ytl = int(a*223/639)
	ybr = int(b*223/639) 
	xtl = int(c*223/479)
	xbr = int(d*223/479)
	'''

	rl = int(data[bb][2])
	ru = int(data[bb][4]) + 1
	cl = int(data[bb][1])
	cu = int(data[bb][3]) + 1
	'''
	
	cropped_im = imarr[rl:ru,cl:cu,:]
	if cropped_im.shape[0]!=0 and cropped_im.shape[1]!=0:
		resized_im = cv2.resize(cropped_im, (224,224), interpolation=cv2.INTER_CUBIC)
	#print(resized_im.shape)

		cropped_list.append(resized_im)
		c = str(idx) + ',' + 'xtl=' + str(xtl) + ',' + 'ytl=' + str(ytl) + ',' + 'xbr=' + str(xbr) + ',' + 'ybr=' + str(ybr) 
		output.append(c)

#print(len(cropped_list))#number of boxes available for training





#load the trained mask and helmet classifier networks
maskclassifier = keras.models.load_model('maskclassifier.h5')
helmetclassifier = keras.models.load_model('helmetclassifier.h5')


#input 
x = np.array(cropped_list[0:3])

#predict output classification probabilities for an image, change the list index to choose one image with cropped_list[0].reshape(1,224,224,3)
mask = maskclassifier.predict(x)
helmet = helmetclassifier.predict(x)


print("mask:" ,mask)
print("helmet:",helmet)

out1 = np.argmax(mask,axis = 1)
out2 = np.argmax(helmet,axis = 1)

maskdict = {0:'invisible',1:'no',2:'wrong',3:'yes'}
helmetdict = {0:'no',1:'yes'}

print(out1)
print(out2)

for m in range(0,mask.shape[0]):
	output[m] =  output[m]  + ',' +'mask=>' + maskdict[out1[m]] + ','+ 'helmet=>' + maskdict[out2[m]]

data = pd.DataFrame(output[0:3])
fpath = 'outtotal.txt'
data.to_csv(fpath, header=None, index=None, sep=' ')