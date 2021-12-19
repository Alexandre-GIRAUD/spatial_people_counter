import os
import tensorflow as tf
import tensorflow_hub as hub
from utils.load_images import load_image_into_numpy_array
import matplotlib.pyplot as plt
from utils.parse_image import parse_image
from utils.create_labelmap import create_labelmap
from utils.parse_model_output import class_counter, draw_boxes
import matplotlib.image 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t","--threshold", help="threshold for validatiing a model prediction",default=0.8)
args = parser.parse_args()

#GLOBAL VARIABLES
IMG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'images')
img_name = "president1.jpg"
#detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
detector = tf.keras.models.load_model(r"C:\Users\Giraud-Alexandre\Desktop\Object_Detection_For_Data_Centers\Spatial_people_counter\centernet_resnet101v1_fpn_512x512_1.tar\centernet_resnet101v1_fpn_512x512_1")
image_array = load_image_into_numpy_array(os.path.join(IMG_PATH,img_name))
print(image_array.shape)

#Prediction
detector_output = detector(image_array)

#Parsed image
label_map = create_labelmap("mscoco_label_map.pbtxt")

logs = class_counter(detector_output,label_map,args.threshold)

try:
    nb_person = logs["person"]
except:
    nb_person=0

print("Sur cette photo il y'a {} personnes".format(nb_person))

#Draw boxes
image_with_boxes = draw_boxes(detector_output,image_array,label_map,args.threshold)
plt.figure()
plt.imshow(image_with_boxes[0])
matplotlib.image.imsave(f"outputs/{img_name[:-4]}.png",image_with_boxes[0])