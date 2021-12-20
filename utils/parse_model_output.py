from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def class_counter(model_output,label_map,threshold):
    """
    Write a log based on the CNN's output
    Inputs:
        model_output, label_map
    Output:
        dict(class:nb_detected)
    """
    logs = {}
    detected_class = model_output["detection_classes"][model_output["detection_scores"]>threshold] #A prediction is validated if its probability is above threshold
    for c in detected_class.numpy():
        try:
            logs[label_map[c]["name"]]+=1
        except:
            logs[label_map[c]["name"]]=1
    return logs

def draw_boxes(model_output,image,label_map,threshold):
    """
    Take an image as input and draw the box over the detected objects
    Inputs:
        model_output,label_map,image,threshold
    Outputs:
        image with the bounding box
    """

    image_with_box = image.copy()
    height = image_with_box.shape[1]
    width = image_with_box.shape[2]
    detected_class = model_output["detection_classes"][model_output["detection_scores"]>threshold].numpy()
    boxes = model_output["detection_boxes"][model_output["detection_scores"]>threshold].numpy()
    for i in range(len(detected_class)):
        name = label_map[detected_class[i]]["name"]
        #draw box
        ymin,xmin,ymax,xmax = boxes[i]
        ymin,xmin,ymax,xmax = int(ymin*height),int(xmin*width),int(ymax*height),int(xmax*width)
        image_with_box[0,max(0,ymin-2):ymin+2,xmin:xmax,:] = 0
        image_with_box[0,ymax-2:min(ymax+2,height),xmin:xmax,:] = 0
        image_with_box[0,ymin:ymax,max(0,xmin-2):xmin+2,:] = 0
        image_with_box[0,ymin:ymax,xmax-2:min(xmax+2,width),:] = 0
    
    return image_with_box[0]

def save_image_with_boxes_and_classes(model_output,image_with_box,label_map,threshold,path):
    """
    Take an image as input and draw class name and proba in boxes
    Inputs:
        model_output,label_map,image (height,width,channels) ,threshold
    Outputs:
        image with the bounding box
    """

    height = image_with_box.shape[0]
    width = image_with_box.shape[1]
    detected_class = model_output["detection_classes"][model_output["detection_scores"]>threshold].numpy()
    boxes = model_output["detection_boxes"][model_output["detection_scores"]>threshold].numpy()
    plt.figure()
    plt.imshow(image_with_box)
    for i in range(len(detected_class)):
        name = label_map[detected_class[i]]["name"]
        proba = model_output["detection_scores"][model_output["detection_scores"]>threshold].numpy()[i]
        txt = name + " : " + str(np.round(proba,2))
        #draw box
        ymin,xmin,ymax,xmax = boxes[i]
        ymin,xmin,ymax,xmax = int(ymin*height),int(xmin*width),int(ymax*height),int(xmax*width)
        plt.text(xmin+10,ymin+40,txt,fontdict={"c":"red"},bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'))
        plt.axis('off')
    
    plt.savefig(path)