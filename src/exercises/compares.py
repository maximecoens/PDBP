import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cv2
import os


def compare(inputFrame, current_state, duration_states, reps, count_reps, exercise):
  # kijken welke oefening met switch
  if exercise == 'upperhand_bicep_curl':
    return compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps, count_reps)
  else:
    return compare_general(inputFrame, current_state, duration_states, reps, count_reps, exercise)
  
keypoints_input = []

def compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps, count_reps):

  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  
  # Feedback for each wrong body position via keypoint detection
  feedback = {0: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "", "", "", "", "", "", ""],
              1: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""],
              2: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "", "", "", "", "", "", ""],
              3: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""],
              4: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "", "", "", "", "", "", ""]}
  
  # TODO: unicodeescape => codec cant decode bytes

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load('src/exercises//upperhand_bicep_curl_delta.npy') 
  files = os.listdir(f'src/screenshots//upperhand_bicep_curl')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join("src/screenshots//upperhand_bicep_curl", image) for image in images]

  # Show first jpg
  if current_state == 0:
    image = cv2.imread(correct_ex_jpg[current_state])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Upperhand bicep curl", image)

  scores_current = [0] * 20
  scores_next = [0] * 20

  coords = np.load(f'src\\exercises\\upperhand_bicep_curl_keypoints_focus.npy')
  for coord in coords:

    # Determine score of current state
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord])

    # Determine score of next state
    scores_next[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]))
    np.seterr(divide='ignore')
    scores_next[coord] = np.arctanh(scores_next[coord])

  wrong_position = False
  if current_state != 0:
    wrong_position = False
    for i in range(5, 11):
      if scores_current[i] <= 1.6:
        wrong_position = True
  
  if wrong_position and current_state != 0:
    duration_states += 1

  # If there is more than 5 seconds between 2 states, the exercise fails.
  if duration_states >= 5:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()

  bt_high = len([num for num in scores_current[5:11] if num > 3.8])
  bt_mid = len([num for num in scores_current[5:11] if num > 2.2])
  bt_min = len([num for num in scores_current[5:11] if num > 1.6])
  
  if current_state == 0 and (bt_high >= 1 and bt_mid >= 2 and bt_min >= 3) \
    or (bt_high >= 3 and bt_mid >= 4 and bt_min >= 6) \
    or (np.average(scores_current) <= np.average(scores_next) and ((bt_high >= 2 and bt_mid >= 3 and bt_min >= 6))):
    
    print("GREAT: This movement was executed perfectly!!")
    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Upperhand bicep curl", image)

    duration_states = 0
    current_state += 1

    if current_state == len(correct_ex_jpg) and reps + 1 == count_reps:
      # show last frame before ending.
      time.sleep(1)

  if current_state == len(correct_ex_jpg):
    reps += 1
    current_state = 0


  return current_state, duration_states, reps


def compare_general(inputFrame, current_state, duration_states, reps, count_reps, exercise):

  # Feedback for each wrong body position via keypoint detection
  feedback = "ATTENTION: The movement was not executed correctly. Check the image!!"
  
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  score_current = 0

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load(f'src/exercises//{exercise}_delta.npy') 
  files = os.listdir(f'src/screenshots//{exercise}')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join(f"src/screenshots//{exercise}", image) for image in images]

  # Show first jpg
  if current_state == 0:
    image = cv2.imread(correct_ex_jpg[current_state])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Movenet model", image)

  scores_current = [0] * 20
  scores_next = [0] * 20

  for coord in range(17):
    
    # Determine score of current state
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord] / 17)

    # Determine score of next state
    scores_next[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]))
    np.seterr(divide='ignore')
    scores_next[coord] = np.arctanh(scores_next[coord])

  wrong_position = False
  if current_state != 0:
    wrong_position = False
    for i in range(17):
      if scores_current[i] <= 1.6:
        wrong_position = True
  
  if wrong_position and score_current != 0:
    print(feedback)
    duration_states += 1

  # If there is more than 5 seconds between 2 states, the exercise fails.
  if duration_states >= 5:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()   

  bt_high = len([num for num in scores_current[5:11] if num > 3.8])
  bt_mid = len([num for num in scores_current[5:11] if num > 2.2])
  bt_min = len([num for num in scores_current[5:11] if num > 1.6])
  
  if current_state == 0 and (bt_high >= 1 and bt_mid >= 2 and bt_min >= 3) \
    or (bt_high >= 3 and bt_mid >= 4 and bt_min >= 6) \
    or (np.average(scores_current) <= np.average(scores_next) and ((bt_high >= 2 and bt_mid >= 3 and bt_min >= 6))):
    
    print("GREAT: This movement was executed perfectly!!")
    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Movenet model", image)

    duration_states = 0
    current_state += 1
    print("GREAT: This movement was executed perfectly!!")

    if current_state == len(correct_ex_jpg) and reps + 1 == count_reps:
      # show last frame before ending.
      time.sleep(1)

  if current_state == len(correct_ex_jpg):
    reps += 1
    current_state = 0

  return score_current, duration_states, reps