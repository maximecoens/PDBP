import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from models.movenet_model2 import predict_movenet_for_video, predict_movenet_for_webcam
import re


print("***********************************************************************")
print("                                                                       ")
print("                 WELCOME TO POSE DETECTION APPLICATION                 ")
print("                                                                       ")
print("       An application born from my Bachelor's thesis at HoGent.        ")
print("                                                                       ")
print("Name:   Maxime Coens                                                   ")
print("Title:  Optimization of body movements through integration of pose     ")
print("        detection models in kinesiology practices. An examination of   ")
print("        potential efficiency gains for kinesiologists.                 ")
print("Date:   May 2024                                                       ")
print("                                                                       ")
print("***********************************************************************")
print("                                                                       ")

print("CHOOSE ACION:")
print("1. Choose Exercise")
print("2. Upload new Exercise")
print("3. Quit")

def choose_ex():
    print("Exercises: ")
    exercises = np.load(f'src\exercises\/exercises.npy')
    for i, ex in enumerate(exercises):
        ex = re.sub("_", " ", ex.capitalize())
        print(i + 1, f". {ex}")
    exercise = int(input())
    if exercise > 0 and exercise < len(exercises):
        print("How many repetitions: ")
        count_reps = int(input())
        # TODO: moet dit iets teruggeven?
        output_keypoints = predict_movenet_for_webcam(exercises[exercise - 1], count_reps)
    else:
        print("INVALID OPTION")
        choose_ex()

def upload_new():

    print("Upload new exercise.")
    print("Name of exercise: ")
    name_ex = str(input())
    name_ex = re.sub(" ", "_", name_ex.lower())
    exercises = np.load('src\exercises\exercises.npy')
    exercises = np.append(exercises, [name_ex])
    np.save(f'src\exercises\exercises.npy', exercises)

    print("Path to video:")
    video_path = str(input())

    print("How many seconds between 2 correct body positions: ")
    delta = float(input())
    delta = round(delta * 30)

    print("Which keypoints are needed to focus on while detecting the body position?")
    print("0. All keypoints!")
    print("1. Nose")
    print("2. Left eye")
    print("3. Right eye")
    print("4. Left ear")
    print("5. Right ear")
    print("6. Left shoulder")
    print("7. Right shoulder")
    print("8. Left elbow")
    print("9. Right elbow")
    print("10. Left wrist")
    print("11. Right wrist")
    print("12. Left hip")
    print("13. Right hip")
    print("14. Left knee")
    print("15. Right knee")
    print("16. Left ankle")
    print("17. Right ankle")
    print("Enter keypoint numbers with space in between: ")
    ex_keypoints = str(input())
    # TODO: hoe doorgeven aan compare? => via nmpy bijhouden, kan sws maar is dit best? => zo kan feedback wel
    # TODO: kan dit voor specifieke feedback? => niet beginnen kutten in lijsten
    if ex_keypoints == "0":
        ex_keypoints = [number for number in range(0,17)]
    else:
        ex_keypoints = [int(number) - 1 for number in ex_keypoints.split(" ")]


    # Gather and save keypoints
    output_keypoints = predict_movenet_for_video(video_path, name_ex, delta)
    np.save(f'src\exercises\{name_ex}.npy', output_keypoints)
    np.save(f'src\exercises\{name_ex}_delta.npy', output_keypoints[::delta])
    print(f"The new exercise {name_ex} was successfully uploaded!")



option = int(input())
match option:
    case 1:
        choose_ex()
    case 2:
        upload_new()
    case 3:
        # TODO
        print("QUIT")
        quit()