import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from models.movenet_model2 import predict_movenet_for_video, predict_movenet_for_webcam


print(71*'*')
print("                                                                       ")
print("                 WELCOME TO POSE DETECTION APPLICATION                 ")
print("                                                                       ")
print("       An application born from my Bachelor's thesis at HoGent.        ")
print("                                                                       ")
print("Name:   Maxime Coens                                                   ")
print("Title:  Optimalisatie van lichaamsbewegingen door integratie van pose  ")
print("        detection modellen in kinesiologische praktijken. Een onderzoek")
print("        naar de potentiële efficiëntievergroting voor kinesisten.      ")
print("Date:   May 2024                                                       ")
print("                                                                       ")
print(71*'*')
print("                                                                       ")

print("CHOOSE ACION:")
print("1. Choose Exercise")
print("2. Upload new Exercise")
print("3. Quit")

def choose_ex():
    print("Exercises: ")
    print("1. Upperhand bicep curl")
    exercise = int(input())
    match exercise:
        case 1:
            output_keypoints = predict_movenet_for_webcam("upperhand_bicep_curl")
        case _:
            print("INVALID OPTION")

def upload_new():
    print("Upload new exercise.")
    print("Name of Exersice:")
    # TODO => maak generiek
    #name_ex = str(input())
    print("Path to video:")
    video_path = str(input())
    output_keypoints = predict_movenet_for_video(video_path, "upperhand_bicep_curl", "new")

option = int(input())
match option:
    case 1:
        choose_ex()
    case 2:
        upload_new()
    case 3:
        # TODO
        print("QUIT")


# webcam opzetten + nodige dingen doen ter voorbereiding oefening.
#video_path = "src\/testdata\JuisteBewNaud.gif"

""" # feedback via oefening
if exercise == "upperhand_bicep_curl":
    #output_keypoints = predict_movenet_for_video(video_path, "upperhand_bicep_curl")
    output_keypoints = predict_movenet_for_webcam("upperhand_bicep_curl") """




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
