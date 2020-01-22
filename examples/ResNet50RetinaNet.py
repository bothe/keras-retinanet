#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, '../')

from examples.utils import sort_and_plot, load_pb_model
from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# use this to change which GPU to use
# set the modified tf session as backend in keras
gpu = 0
setup_gpu(gpu)

# ## Load RetinaNet models
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

start = time.time()
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
print("Load normal model - processing time: ", time.time() - start)

if not os.path.exists('model.pb'):
    # Save model for TF Inference
    model.save('model.pb')
    pb_model = load_pb_model()
else:
    pb_model = load_pb_model()

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model_converted = models.convert_model(model) # Not working
# print(model.summary())

# ## Run detection on example
# load image
image = read_image_bgr('000000397133.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# pre-process image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# prediction with normal model on image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("prediction with normal model processing time: ", time.time() - start)

sort_and_plot(boxes, scores, labels, draw, scale, threshold=0.4)

# prediction with .pb model on  image
start = time.time()
boxes, scores, labels = pb_model.predict_on_batch(np.expand_dims(image, axis=0))
print("prediction with .pb model processing time: ", time.time() - start)

sort_and_plot(boxes, scores, labels, draw, scale, threshold=0.4)
