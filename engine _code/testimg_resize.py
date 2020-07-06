import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
import cv2
dirName = r'imagestest/'
imagelist = []
for (dirpath, dirnames, filenames) in os.walk(dirName):
	
	for file in filenames:
		imagelist.append(file)
		
		print(file)

dirName = 'imagestest/';#images folder path


for img in imagelist:
	
	
	filepath = dirName + img 
	imd1 = image.load_img(filepath)
	imarr1 = np.array(imd1)
	#indexing.append(index) 
	resized_im = cv2.resize(imarr1, (640,480), interpolation=cv2.INTER_CUBIC)
	
	writepath =  'keras-frcnn-master/test_images/' + img
	image.save_img(writepath,resized_im)