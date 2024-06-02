import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from models.movenet_model2 import predict_movenet_for_video, predict_movenet_for_webcam
import re
import tensorflow as tf



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

def show_info():
    print("                                                                       ")
    print("************************ Movenet Lightning f16 ************************")
    print("Deze versie is geschikt voor toepassingen met beperkte rekenkracht,")
    print("zoals mobiele apparaten. Hier ligt de nadruk op snelheid in plaats")
    print("van precisie, waardoor dit model geschikt is voor het detecteren van")
    print("eenvoudige bewegingen.")
    print("***********************************************************************")
    print("                                                                       ")
    print("************************* Movenet Thunder f16 *************************")
    print("In tegenstelling tot het Lightning-model kan deze versie hogere")
    print("rekenkracht aan en richt het zich op precisie. Dit model is geschikt")
    print("voor gedetailleerde analyses van complexe bewegingen, zoals")
    print("dansbewegingen.")
    print("***********************************************************************")
    print("                                                                       ")
    print("*********************** Movenet Lightning int8 ************************")
    print("Deze versie is ontworpen voor uiterst snelle toepassingen met zeer")
    print("beperkte rekenkracht.")
    print("***********************************************************************")
    print("                                                                       ")
    print("*********************** Movenet Thunder int8 **************************")
    print("Dit model biedt een balans tussen nauwkeurigheid en snelheid, met")
    print("minder rekenkracht dan de Float16-versies.")
    print("***********************************************************************")
    print("                                                                       ")
    print("************** Lite-model Movenet singlepose Lightning 3 **************")
    print("Dit model maakt gebruik van volledige floatprecisie, wat resulteert in")
    print("hogere nauwkeurigheid dan de half-precisie modellen (Float16). Dit")
    print("model is geschikt voor toepassingen waar rekenkracht geen beperking ")
    print("vormt, zoals wetenschappelijk onderzoek.")
    print("***********************************************************************")
    print("                                                                       ")

def choose_ex():
    exercises = np.load(f'src\\exercises\\exercises.npy')
    exercise = -1
    while (exercise <= 0 or exercise > len(exercises)):
        print("Exercises: ")
        for i, ex in enumerate(exercises):
            ex = re.sub("_", " ", ex.capitalize())
            print(f"{i + 1}. {ex}")
        exercise = int(input())

    count_reps = 0
    while (count_reps <= 0):
        print("How many repetitions: ")
        count_reps = int(input())
    
    model = 0
    while (model >= 6 or model < 1):
        print("Which Movenet model do you want to use?")
        print("1. movenet_lightning_f16")
        print("2. movenet_thunder_f16")
        print("3. movenet_lightning_int8")
        print("4. movenet_thunder_int8")
        print("5. lite-model_movenet_singlepose_lightning_3")
        print("6. Model info")
        model = int(input())
        if model == 6:
            show_info()

    predict_movenet_for_webcam(exercises[exercise - 1], count_reps, model)

def upload_new():

    print("Upload new exercise.")
    print("Name of exercise: ")
    name_ex = str(input())
    name_ex = re.sub(" ", "_", name_ex.lower())
    exercises = np.load('src\\exercises\\exercises.npy')
    if name_ex in exercises:
        print("Exercise already uploaded or name of exercise already in use! Run program again.")
        quit()
    exercises = np.append(exercises, [name_ex])
    np.save(f'src\\exercises\\exercises.npy', exercises)

    print("Path to video:")
    video_path = str(input())

    model = 0
    while (model > 6 or model < 1):
        print("Which Movenet model do you want to use?")
        print("1. movenet_lightning_f16")
        print("2. movenet_thunder_f16")
        print("3. movenet_lightning_int8")
        print("4. movenet_thunder_int8")
        print("5. lite-model_movenet_singlepose_lightning_3")
        print("6. Model info")
        model = int(input())
        if model == 6:
            show_info()


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
    
    if ex_keypoints == "0":
        ex_keypoints = [number for number in range(0,17)]
    else:
        ex_keypoints = [int(number) - 1 for number in ex_keypoints.split(" ")]
    np.save(f'src\\exercises\\{name_ex}_keypoints_focus.npy', ex_keypoints)


    # Gather and save keypoints
    output_keypoints = predict_movenet_for_video(video_path, name_ex, delta, model)
    #np.save(f'src\\exercises\\{name_ex}.npy', output_keypoints) # TODO: weg + focus anders oplossen?
    np.save(f'src\\exercises\\{name_ex}_delta.npy', output_keypoints[::delta])
    print(f"The new exercise {name_ex} was successfully uploaded!")



option = int(input())
match option:
    case 1:
        choose_ex()
    case 2:
        upload_new()
    case 3:
        print("QUIT")
        quit()