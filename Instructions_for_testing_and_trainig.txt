The engine is implemented using Keras v2.2.0 and tensorflow v1.8.0
The above versions are recommended while running the engine to avoid bugs in the later versions.
Dependancies : keras,opencv,python,sklearn,h5py,numpy,os,pandas,xml,scipy


STEPS TO TRAIN THE ENGINE
-------------------------


===>Store train images in the folder :engine->images

===>Store annotation file in xml format as: engine->annotations.xml

===>steps to train FRCNN OBJECT DETECTOR:

=>Run the file :engine->read_xml.py  
o This file reads the train images and parses annotations.xml
o Writes the resized images in : engine->keras-frcnn-master->train_images
o Writes the FRCNN labels in : engine->keras-frcnn-master->annotate.txt

=>Run the file :engine->keras-frcnn-master->train_frcnn.py using 'python train_frcnn.py -o simple -p annotate.txt'
o This file trains the FRCNN network to detect head bounding boxes from images
o To change the number of epochs(default 2000), go to the train_frcnn.py file and change the num_epochs parameter accordingly.
o Every time the model sees an improvement, the weights of that particular epoch will be saved as : engine->keras-frcnn-master->model_frcnn.hdf5

===>steps to train MASK CLASSIFIER NETWORK:

=>Run the file :engine->mask_classifier_train.py 
o this file reads the images and annotations from engine->images and engine->annotations.xml respectively
o crops the bounding boxes from images using box coordinates
o trains the network with cropped and resized bounding boxes
o the trained model is saved as : engine->maskclassifier.h5
o to change batch size(default 40), epochs(default 100), validation split(default 0.3) change the parameters in line:216 of the file 


===>steps to train SAFETY HELMET CLASSIFIER NETWORK:

=>Run the file :engine->safety_helmet_classifier_train.py
o this file reads the images and annotations from engine->images and engine->annotations.xml respectively
o crops the bounding boxes from images using box coordinates
o trains the network with cropped and resized bounding boxes
o the trained model is saved as : engine->helmetclassifier.h5
o to change batch size(default 40), epochs(default 100), validation split(default 0.3) change the parameters in line:217 of the file 



STEPS TO RUN THE ENGINE ON UNSEEN IMAGES
----------------------------------------

===>Store the unseen images in the folder :engine->imagestest

===>First run the FRCNN OBJECT DETECTOR:

=>Run the file :engine->testimg_resize.py
o this file reads the test images from :engine->imagestest
o writes the resized files to :engine->keras-frcnn-master->test_images

=>Run the file :engine->keras-frcnn-master->test_frcnn.py using 'python test_frcnn.py -p test_images'
o this file reads the test images from :engine->keras-frcnn-master->test_images
o this file loads the trained FRCNN model from : engine->keras-frcnn-master->model_frcnn.hdf5
o Runs the test images through the FRCNN network
o writes the output bounding box labels for detected 'head' bounding boxes to : engine->frcnnoutput.csv
o the format of the bounding box labels is : 0.jpg,xtl,ytl,xbr,ybr,head  


===>NOW run the MASK and SAFETY HELMET CLASSIFIERS:

=>Run the file :engine->complete_classifier_network.py
o this file reads and resizes the test images from :engine->imagestest
o reads the FRCNN object detector output bounding box labels from :engine->frcnnoutput.csv
o crops the bounding boxes from resized test images using resized box coordinates
o loads the trained mask classifier model from : engine->maskclassifier.h5
o loads the trained safety helmet classifier model from : engine->helmetclassifier.h5
o Runs the cropped and resized bounding box images thruogh the classifier networks
o writes the output labels to : engine->outtotal.txt
o the format of the output labels is : 0.jpg,xtl=174,ytl=56,xbr=220,ybr=126,mask=>yes,helmet=>invisible



