
def class_counter(model_output,label_map,threshold):
    """
    Write a log based on the CNN's output
    Input:
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


        
