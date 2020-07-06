# Inference-Engine
A two stage inference engine that detects whether workers are wearing safety helmets and masks from RGB images.
I have divided the task of detecting if a person is wearing a mask and safety helmet into two separate tasks:

Task 1: First, an object detector network is used to predict bounding boxes
containing heads in the image.

Task 2: Next, two parallel image classifier networks are used to classify the
detected head bounding boxes into 4 classes for the presence of mask(yes, no,
invisible, wrong) and 2 classes for the presence of safety helmets(yes, no).


So, the engine consists of two stages:

Stage-1: Object detector network(Faster RCNN.) that takes an RGB image as
input and produces bounding box locations for heads in the image. The object classifier
branch of faster Rcnn classifies each ROI into three classes : head, nonhead and
background. The object bounding box regressor branch produces 4 coordinates xtl, ytl,
xbr, ybr specifying the bounding box location.

Stage-2 : 2 parallel Image classifier networks for mask and helmet.

Classifier 1(mask classifier) : Takes the bounding boxes labeled as head from the
faster Rcnn detector and crops the bounding boxes from the image and takes them as
input. The output has 4 classes( depending on values present in annotation) viz
(invisible, no, wrong, yes) denoting the presence of masks in the image. The architecture
summary is shown below. Note that the final dense layer has 4 classes.



Classifier 2(safety helmet classifier) :Takes the bounding boxes labeled as head
from the faster Rcnn and crops the bounding boxes from the image and takes them as
input. The output has 2 classes( depending on values present in annotation) viz (yes, no)
denoting the presence of safety helmets in the image. The architecture summary is shown
below. Note that the final dense layer has 2 classes.



The dataset with images and annotations can be found here:
https://drive.google.com/drive/folders/1TFjsTFspmtyViBOmuCo-tBbXZ72GhVuG?usp=sharing


