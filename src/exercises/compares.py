import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cv2
import os


def compare(inputFrame, current_state, duration_states, reps, exercise):
  # kijken welke oefening met switch
  if exercise == 'upperhand_bicep_curl':
    return compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps)
  else:
    return compare_general(inputFrame, current_state, duration_states, reps, exercise)
  
keypoints_input = []

def compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps):

  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  
  # Feedback for each wrong body position via keypoint detection
  feedback = {0: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "", "", "", "", "", "", ""],
              1: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""],
              2: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "", "", "", "", "", "", ""],
              3: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""],
              4: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "", "", "", "", "", "", ""]}
  
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  # TODO: time delta werken, tussen current state en next state ook meer dan 10 frames?
  # states = np.load('src\exercises//upperhand_bicep_curl.npy') # _delta ? zie general
  score_current = 0

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load('src/exercises//upperhand_bicep_curl.npy') 
  files = os.listdir(f'src/screenshots//upperhand_bicep_curl')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join("src/screenshots//upperhand_bicep_curl", image) for image in images]

  # TODO: Testen voor tonen van verschillende foto's + werkt dit? => testen
  # TODO: checken voor specifieke feedback
  scores_current = [0] * 20

  for coord in range(5, 11):

    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord])

  wrong_position = False
  for i in range(5, 11):
    if scores_current[i] <= 2.5:
      wrong_position = True
      print(feedback[current_state][i])
  
  if wrong_position and score_current != 0:
    duration_states += 1
  #TODO: check hoelang hier tussen moet zitten, om de hoeveel seconden roept hij compare op en hoelang voor wrong state optellen
  if duration_states >= 10:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()

  if all(score > 2.5 for score in scores_current):

    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Correct exercise", image)
    # cv2.waitKey(0)

    duration_states = 0
    current_state += 1
    print("GREAT: This movement was executed perfectly!!")
  
  if current_state == len(correct_ex_jpg) - 1:
    reps += 1
    current_state = 0

  return score_current, duration_states, reps


## TODO: kijken of het werkt voor een general werking met 17 keypoints! => al zittend niet mogelijk
## mogelijks keypoints weergeven?
## TODO: reps

def compare_general(inputFrame, current_state, duration_states, reps, exercise):

  # Feedback for each wrong body position via keypoint detection
  feedback = "ATTENTION: The movement was not executed correctly. Check the image!!"
  
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  # TODO: time delta werken, tussen current state en next state ook meer dan 10 frames?
  # states = np.load('src\exercises//upperhand_bicep_curl.npy') # _delta ? zie general
  score_current = 0

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load(f'src/exercises//{exercise}_delta.npy') 
  files = os.listdir(f'src/screenshots//{exercise}')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join(f"src/screenshots//{exercise}", image) for image in images]

  scores_current = [0] * 20

  ## TODO: Testen voor tonen van verschillende foto's + werkt dit?
  # TODO: werken met default 17 keypoints? of vragen welke
  for coord in range(17):
    
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord] / 17)

  wrong_position = False
  for i in range(17):
    if scores_current[i] <= 2.5:
      wrong_position = True
      print(feedback)
  
  if wrong_position and score_current != 0:
    duration_states += 1
  # TODO: check hoelang hier tussen moet zitten, om de hoeveel seconden roept hij compare op en hoelang voor wrong state optellen
  if duration_states >= 10:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()   

  if all(score > 2.5 for score in scores_current):

    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Correct exercise", image)
    # cv2.waitKey(0)

    duration_states = 0
    current_state += 1
    print("GREAT: This movement was executed perfectly!!")
  
  if current_state == len(correct_ex_jpg) - 1:
    reps += 1
    current_state = 0

  return score_current, duration_states, reps