#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[30]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

import cv2

# import win32com.client as comclt
# wsh= comclt.Dispatch("WScript.Shell")

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

inputVideoPath = './first-cut.mp4'
outputVideoPath = 'output.avi'


def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

def draw_lot(img, coord):
	for i in range (len(coord)-1):
		cv2.line(img, coord[i], coord[i+1], (255, 0, 0), 5) # (start x,y) , (end x,y)
# 		cv2.putText(img, str(i+1), coord[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 
# 						(255, 0, 0), 3, cv2.LINE_AA)

	cv2.line(img, coord[-1], coord[0], (255, 0, 0), 5)
# 	cv2.putText(img, str(len(coord)), coord[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, 
# 					(255, 0, 0), 3, cv2.LINE_AA)

def pos_angle(num_frames):
	pos = -1
	angle = -1
	if num_frames >= 240 and num_frames <= 540:
		pos = 0
		angle = 0
	if num_frames >= 630 and num_frames <= 840:
		pos = 0
		angle = 1
	
	return pos, angle

def coordinates_draw(frame):
    if pos == 0 and angle == 0:
        x1, y1 = 305, 665	
        x2, y2 = 520, 405
        coord = [(x1,y1),(x2,y2)]
        draw_lot(frame, coord)

        x1, y1 = 520, 405	
        x2, y2 = 695, 210
        coord = [(x1,y1),(x2,y2)]
        draw_lot(frame, coord)

    elif pos == 0 and angle == 1:
        x1, y1 = 510, 420
        x2, y2 = 670, 235
        x3, y3 = 450, 200
        x4, y4 = 265, 375
        lot1 = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        draw_lot(frame, lot1)

        x1, y1 = 900, 470
        x2, y2 = 920, 250
        x3, y3 = 670, 235
        x4, y4 = 510, 420
        lot2 = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        draw_lot(frame, lot2)


## TODO: warning text hard coded

num_frames = 0

## TODO: videoCapture
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = initializeVideoWriter(video_width, video_height, videoStream)

while True:
    # image = cv2.imread(IMAGE_DIR + '/690.jpg')
    (grabbed, image) = videoStream.read()
    if not grabbed:
        break
    num_frames += 1
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = skimage.io.imread(IMAGE_DIR + '/690.jpg')

    # Run detection
    results = model.detect([img_rgb], verbose=0)
    # print(results)
    r = results[0]

    pos, angle = pos_angle(num_frames)
    coordinates_draw(img_rgb)

    # filter out low scoring items
    num_items = 0
    for i in r['scores']:
        if i >= 0.8:
            num_items += 1
    # print(num_items)

    # r['rois'] = r['rois'][:num_items]
    # r['class_ids'] = r['class_ids'][:num_items]
    # r['scores'] = r['scores'][:num_items]
    # print(r['scores'])


    items = np.zeros((num_items))
    # print(items)
    items_in = np.zeros((num_items))

    import time
    start_time = time.time()

    row_num, px_num = 0, 0

    # 670 * 1192 * number of items
    for row in r['masks']:
    #     print(row_num)
        if np.any(row): # comment out this for box reduction
            for px in row:
        #             print (px)
                # px = px[:num_items] # for box reduction
                if np.any(px):
        #                 print (px)
                    for i in range(len(px)):
                        if i>=num_items:
                            px[i] = False
                        elif px[i]:
                            items[i] += 1
                        
                        
                px_num += 1
        px_num = 0
        row_num += 1
    #     print(px_num, row_num)
    # print("masked pixels of each item:", items)
    # print("masked pixels inside lot:", items_in)

    # ratio = items_in[1]/items[1]
    # print(ratio)

    print("time:", time.time()-start_time)
                    

    masked_img = visualize.display_instances(img_rgb, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
    
    # wsh.SendKeys("q")
    writer.write(masked_img)


writer.release()
videoStream.release()
