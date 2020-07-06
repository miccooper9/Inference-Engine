import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
import cv2


tree = ET.parse('annotations.xml')
root = tree.getroot()

numinum = []
boxlabels = []


#generate the bounding box labels from xml file
for im in root.findall("image"):
	
	
	
	inum = im.get('id')
	inumjpg = inum + '.jpg'
	numinum.append(inumjpg)
	print(inum)
	for box in im.findall("box"):
		
		label = box.get('label')
		
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
		blist.append(label)
			
		boxlabels.append(blist)
				
print(len(boxlabels))


imagelist = set()#set of image ids in images folder
dirName = r'images/';#images folder path

#generating imagelist
for (dirpath, dirnames, filenames) in os.walk(dirName):
	
	for file in filenames:
		imagelist.add(file)
		
		print(file)
   
				
print(len(numinum))
print(len(imagelist))

#there are 591 images in image folder and annotations for 527 images

#choosing the largest common subset between the images and labels

for ii in numinum:
	if not (ii in imagelist):
		numinum.remove(ii)


#number of images with labels available for training
print(len(numinum))



dirName = 'images/';#images folder path


actualboxes = boxlabels#boxes available for training

#generating actualboxes
resized_list =[]#list of boxes resized
'''
for img in numinum:
	
	imstring = img.replace(".jpg","")
	index = int(imstring)
	#indexing.append(index)

	for bx in boxlabels:
		if(bx[0]==index):
			actualboxes.append(bx)

'''

#print(actualboxes)
print(len(actualboxes))#number of boxes available for training

#resize images to the same size 640 x 480
for img in numinum:
	
	
	filepath = dirName + img 
	imd1 = image.load_img(filepath)
	imarr1 = np.array(imd1)
	#indexing.append(index) 
	resized_im = cv2.resize(imarr1, (640,480), interpolation=cv2.INTER_CUBIC)
	resized_list.append(resized_im)
	writepath =  'keras-frcnn-master/'+'train_images/' + img
	image.save_img(writepath,resized_im)

annotationtxt =[]

#generating cropped_list
for bb in actualboxes:
	idx = bb[0]
	filepath = dirName + str(idx) + '.jpg'
	
	imd = image.load_img(filepath)
	imarr = np.array(imd)
	writepath = 'train_images/' + str(idx) + '.jpg'
	y1 = int(bb[2])
	y2 = int(bb[4])
	x1 = int(bb[1])
	x2 = int(bb[3]) 
	bb[0] = writepath #str(bb[0]) 
	bb[1]= str(int((x1*(479))/(imarr.shape[1]-1)))
	bb[3]= str(int((x2*(479))/(imarr.shape[1]-1))) 
	bb[2]= str(int((y1*(639))/(imarr.shape[0]-1)))
	bb[4]= str(int((y2*(639))/(imarr.shape[0]-1)))
	if(bb[5]!='head'):
		bb[5]='nothead'
	#print(x1,y1,x2,y2,"====>",bb[1],bb[2],bb[3],bb[4])
	f = bb[0] + ',' + bb[1] + ','+bb[2] + ','+bb[3] + ','+ bb[4] + ',' +bb[5]
	annotationtxt.append(f)

	
	#print(resized_im.shape)
	

print(len(resized_list))#number of boxes available for training





#check the data
boxes = pd.DataFrame(actualboxes)
data = pd.DataFrame(annotationtxt)
boxdata = boxes.values
labels = boxdata[:,5]
print(labels)
print(len(numinum))
print(boxdata.shape)
print(boxes.head())

'''				
data = pd.DataFrame()
data['format'] =  boxes.iloc[:,:]

# as the images are in train_images folder, add train_images before the image name

for i in range(data.shape[0]):
    data['format'][i] = 'resized_images/' + data['format'][i]


# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
	data['format'][i] = data['format'][i] + ',' + str(boxes.iloc[i,1]) + ',' + str(boxes.iloc[i,4]) + ',' + str(boxes.iloc[i,3]) + ',' + str(boxes.iloc[i,2]) + ',' + boxes.iloc[i,5]
'''
#generate annotation for Faster rcnn
dirName = r'keras-frcnn-master/';
fpath = dirName + 'annotate.txt'
data.to_csv(fpath, header=None, index=None, sep=' ')
			
	