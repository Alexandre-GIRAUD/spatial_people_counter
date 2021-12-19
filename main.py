import os
import tensorflow as tf
import tensorflow_hub as hub
from utils.load_images import load_image_into_numpy_array
import matplotlib.pyplot as plt
from utils.parse_image import parse_image
from utils.create_labelmap import create_labelmap
from utils.parse_model_output import class_counter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t","--threshold", help="threshold for validatiing a model prediction",default=0.8)
args = parser.parse_args()

#GLOBAL VARIABLES
IMG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'images')

#detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
detector = tf.keras.models.load_model(r"C:\Users\Giraud-Alexandre\Desktop\Object_Detection_For_Data_Centers\Spatial_people_counter\centernet_resnet101v1_fpn_512x512_1.tar\centernet_resnet101v1_fpn_512x512_1")
image_tensor = load_image_into_numpy_array(os.path.join(IMG_PATH,"president1.jpg"))

#Prediction
detector_output = detector(image_tensor)

#Parsed image
label_map = create_labelmap("mscoco_label_map.pbtxt")

logs = class_counter(detector_output,label_map,args.threshold)

print("Sur cette photo il y'a {} personnes".format(logs["person"]))