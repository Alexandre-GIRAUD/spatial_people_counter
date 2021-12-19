import os
import tensorflow as tf
import tensorflow_hub as hub
from utils.load_images import load_image_into_numpy_array
import matplotlib.pyplot as plt
from utils.parse_image import parse_image

#GLOBAL VARIABLES
IMG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'images')

#detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
detector = tf.keras.models.load_model(r"C:\Users\Giraud-Alexandre\Desktop\Object_Detection_For_Data_Centers\Spatial_people_counter\centernet_hourglass_512x512_1.tar\centernet_hourglass_512x512_1")
image_tensor = load_image_into_numpy_array(os.path.join(IMG_PATH,"datacenter1.jpg"))

#print image
plt.figure(figsize=(24,32))
plt.imshow(image_tensor[0])
plt.show()

#Prediction
detector_output = detector(image_tensor)

#Parsed image
