
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
    return image_with_box