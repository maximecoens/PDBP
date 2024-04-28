import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from models.movenet_model2 import predict_movenet_for_video, predict_movenet_for_webcam


print("INFOINFOINFOINFO")
print("CHOOSE EXERCISE: BOVENHANDSE BICEP CURL")

exercise = "upperhand_bicep_curl"

# webcam opzetten + nodige dingen doen ter voorbereiding oefening.
video_path = "src\/testdata\BovCurlCorrect\/fr90graden.jpg"

# feedback via oefening
if exercise == "upperhand_bicep_curl":
    #output_keypoints = predict_movenet_for_video(video_path, "upperhand_bicep_curl")
    output_keypoints = predict_movenet_for_webcam("upperhand_bicep_curl")





""" output_keypoints = [] # mm2.predict_movenet_for_video(video_path)

if output_keypoints is not None:    
    print("Converted to Gif Successfully")

# om de 6 frames bijhouden
framerate = 6
output_keypoints_input = []
for frame in range(0, len(output_keypoints), framerate):
   output_keypoints_input.append(output_keypoints[frame])
print(output_keypoints_input)
"""
