import os
import tensorflow as tf
import tensorflow_hub as hub
from utils.load_images import load_image_into_numpy_array
import matplotlib.pyplot as plt
from utils.parse_image import parse_image
from utils.create_labelmap import create_labelmap
from utils.parse_model_output import class_counter, draw_boxes ,save_image_with_boxes_and_classes
import matplotlib.image 
from datetime import date, datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i","--image",help="path of input of the neuron",required=True)
parser.add_argument("-t","--threshold", help="threshold for validatiing a model prediction",default=0.8)
args = parser.parse_args()


def prediction(path_image,t):
    #GLOBAL VARIABLES

    #detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
    detector = tf.keras.models.load_model(r"C:\Users\Giraud-Alexandre\Desktop\Object_Detection_For_Data_Centers\Spatial_people_counter\centernet_resnet101v1_fpn_512x512_1.tar\centernet_resnet101v1_fpn_512x512_1")
    image_array = load_image_into_numpy_array(path_image)
    #Prediction
    detector_output = detector(image_array)

    #Parsed image
    label_map = create_labelmap("mscoco_label_map.pbtxt")

    logs = class_counter(detector_output,label_map,t)

    try:
        nb_person = logs["person"]
    except:
        nb_person=0
    today = date.today()
    day = today.strftime("%b-%d-%Y")
    
    with open(f"logs/logs_{day}.txt",'w+') as f:
        now = datetime.now()
        h = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write("\n")
        f.write(h + f" : {nb_person} personnes")
        

    #Draw boxes
    image_with_boxes = draw_boxes(detector_output,image_array,label_map,t)

    save_image_with_boxes_and_classes(detector_output,image_with_boxes,label_map,t, path=f"outputs/" + path_image)

if __name__=="__main__":
    prediction(args.image,args.threshold)