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



tree = ET.parse('annotations.xml')
root = tree.getroot()


numinum = []#list of image ids in annotations.xml
boxlabels = []#list of boxes in annotations.xml

#generate numinum and boxlabels from annotations.xml
for im in root.findall("image"):
	
	
	
	inum = im.get('id')
	inumjpg = inum + '.jpg'
	numinum.append(inumjpg)
	print(inum)
	for box in im.findall("box"):
		
		label = box.get('label')
		if(label == "head"):
			xt = box.get('xtl')
			yt = box.get('ytl')
			xb = box.get('xbr')
			yb = box.get('ybr')
			blist = []
			blist.append(int(inum))
			blist.append(float(xt))
			blist.append(float(yt))
			blist.append(float(xb))
			blist.append(float(yb))
			#print(inum,xt,yt,xb,yb)
			for attr in box.findall(".//*[@name='has_safety_helmet']"):
				atb  = attr.get('name')
				val = attr.text
				blist.append(val)
				#print(atb,"====>",val)
			for attr in box.findall(".//*[@name='mask']"):
				atb  = attr.get('name')
				val = attr.text
				blist.append(val)
				#print(atb,"====>",val)
			boxlabels.append(blist)
				
#print(boxlabels)







imagelist = set()#set of image ids in images folder
dirName = r'images/';#images folder path

#generating imagelist
for (dirpath, dirnames, filenames) in os.walk(dirName):
	
	for file in filenames:
		imagelist.add(file)
		
		print(file)
   
				

#there are 591 images in image folder and annotations for 527 images

#choosing the largest common subset between the images and labels
'''
for ii in numinum:
	if not (ii in imagelist):
		numinum.remove(ii)

'''
#number of images with labels available for training

print(len(numinum))



dirName = 'images/';#images folder path


actualboxes = boxlabels#boxes available for training

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

print(actualboxes)
print(len(actualboxes))#number of boxes available for training
print(len(boxlabels))
cropped_list =[]#list of boxes cropped 



#generating cropped_list
for bb in actualboxes:
	idx = bb[0]
	filepath = dirName + str(idx) + '.jpg'
	imd = image.load_img(filepath)
	imarr = np.array(imd)
	rl = int(bb[2])
	ru = int(bb[4]) + 1
	cl = int(bb[1])
	cu = int(bb[3]) + 1
	cropped_im = imarr[rl:ru,cl:cu,:]
	resized_im = cv2.resize(cropped_im, (224,224), interpolation=cv2.INTER_CUBIC)
	#print(resized_im.shape)
	cropped_list.append(resized_im)

print(len(cropped_list))#number of boxes available for training


boxes = pd.DataFrame(actualboxes)
boxdata = boxes.values

print(boxdata[:,5:7])
print(boxdata.shape)

'''
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
print(X_train.shape)

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

'''
inputx = np.array(cropped_list)


a,y = np.unique(boxdata[:,6],return_inverse=True)
print(a)
print(y)
inputy = np_utils.to_categorical(y, 4)
print(inputy)

print(inputy.shape)
print(inputx.shape)
X_train, X_test, y_train, y_test = train_test_split(inputx, inputy, test_size = 0.2)
print(X_test.shape,X_train.shape)
print(y_test.shape,y_train.shape)
'''
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(),[5])],remainder = 'drop')
labels = np.array(ct.fit_transform(boxdata))

#annotations for available boxes
print(labels)
print(labels.shape)
'''





#mask classifier model
inputim = Input(shape =(224,224,3))

resout = ResNet50(weights='imagenet', include_top=False)(inputim)

pool_global_average = GlobalAveragePooling2D(data_format='channels_last')(resout)

drop = Dropout(0.5)(pool_global_average)

dense = Dense(5, activation='softmax')(drop)

drop1 = Dropout(0.5)(dense)

dense1 = Dense(4, activation='softmax')(drop1)

maskclassifier = Model(inputs = inputim, outputs = dense1)


print(maskclassifier.summary())

maskclassifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#train
maskclassifier.fit(X_train, y_train, batch_size = 40, epochs = 100, validation_split=0.3, verbose =1)

#save the model
maskclassifier.save('maskclassifier.h5')
# Predicting the Test set results
#y_pred = sigclass.predict(X_test)