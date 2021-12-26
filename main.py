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
parser.add_argument("-m","--model",help="URL/path to the CNN model used for object detection",default="https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")
args = parser.parse_args()


def prediction(path_image,t,model_path):
    """
    Take an image and return the output of the model + update the logs

    Input:
        path_image
        t: threshold to accept a prediction from the model
        model_path: URL from tensorflow hub or path for a local model
    """
    #Load Object detection model
    if model_path.startswith("https"):
        detector = hub.load(model_path)
    else:
        detector = tf.keras.models.load_model(model_path)
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

        f.write(h + f" : {nb_person} personnes \n")
        

    #Draw boxes
    image_with_boxes = draw_boxes(detector_output,image_array,label_map,t)

    save_image_with_boxes_and_classes(detector_output,image_with_boxes,label_map,t, path=f"outputs/" + os.path.basename(path_image).split(".")[0] + ".png")

if __name__=="__main__":
    prediction(args.image,args.threshold,args.model)