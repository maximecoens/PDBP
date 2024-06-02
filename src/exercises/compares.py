import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cv2
import os


def compare(inputFrame, current_state, duration_states, reps, count_reps, exercise):
  match exercise:
    case 'upperhand_bicep_curl':
      return compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps, count_reps)
    case _:
      return compare_general(inputFrame, current_state, duration_states, reps, count_reps, exercise)
  
keypoints_input = []
def compare_bovenhandsecurl(inputFrame, current_state, duration_states, reps, count_reps):

  """ This procedure is used to compare 2 consecutive states.

      Input: Coordinates of keypoints on inputframes,
             current state of body position,
             duration of the current state,
             number of repetitions executed,
             number of repetitions needed to complete exercise

      Output: Current state of body position after comparison,
              duration of the current state,
              number of repetitions executed
  """
  
  # Feedback for each wrong body position via keypoint detection
  feedback = {0: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 180 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 180 degrees position!", "", "", "", "", "", "", ""],
              1: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""],
              2: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 90 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 90 degrees position!", "", "", "", "", "", "", ""],
              3: ["", "", "", "", "", "", "", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (left arm): Make sure your arms are in a 45 degrees position!", "ATTENTION (right arm): Make sure your arms are in a 45 degrees position!", "", "", "", "", "", "", ""]}

  # paths to correct exercise pictures
  correct_ex_keypoints = np.load('src/exercises//upperhand_bicep_curl_delta.npy')
  files = os.listdir(f'src/screenshots//upperhand_bicep_curl')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join("src/screenshots//upperhand_bicep_curl", image) for image in images]

  # Show first jpg
  if current_state == 0:
    image = cv2.imread(correct_ex_jpg[current_state])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Upperhand bicep curl", image)

  # Check that all keypoints are detected
  coords = np.load(f'src\\exercises\\upperhand_bicep_curl_keypoints_focus.npy')
  print(inputFrame[5:11])
  for coord in coords:
    if inputFrame[coord][2] < 0.5:
      print("Not all keypoints are detected.")
      return current_state, duration_states, reps

  scores_current = [0] * 20
  scores_next = [0] * 20
  for coord in coords:

    # Determine score of current state
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord])

    # Determine score of next state
    scores_next[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]))
    np.seterr(divide='ignore')
    scores_next[coord] = np.arctanh(scores_next[coord])

  print(scores_current[5:11])
  # Give specific feedback based on scores
  wrong_position = False
  for i in coords:
    if scores_current[i] <= 2.25:
      wrong_position = True
      print(feedback[current_state][i])

  diff = np.average(scores_current[5:11]) - np.average(scores_next[5:11])
  if (diff < 0) and not wrong_position:
    print("ATTENTION: Movement was not executed right!")
    wrong_position = True

  print(np.average(scores_current[5:11]))
  print(np.average(scores_next[5:11]))
  if wrong_position and current_state != 0:
    duration_states += 1

  # If there is more than 10 seconds between 2 states, the exercise fails.
  if duration_states >= 10:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()

  if not wrong_position:
    print("GREAT: This movement was executed perfectly!")
    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Upperhand bicep curl", image)

    duration_states = 0
    current_state += 1

    if current_state == len(correct_ex_jpg) and reps + 1 == count_reps:
      # show last frame before ending.
      time.sleep(2)

  # Go to next state
  if current_state == len(correct_ex_jpg):
    reps += 1
    current_state = 0


  return current_state, duration_states, reps


def compare_general(inputFrame, current_state, duration_states, reps, count_reps, exercise):

  """ This procedure is used to compare 2 consecutive states.

      Input: Coordinates of keypoints on inputframes,
             current state of body position,
             duration of the current state,
             number of repetitions executed,
             number of repetitions needed to complete exercise,
             name of exercise

      Output: Current state of body position after comparison,
              duration of the current state,
              number of repetitions executed
  """  

  # Feedback for each wrong body position via keypoint detection
  feedback = "ATTENTION: The movement was not executed correctly. Check the image!!"

  # paths to correct exercise pictures
  correct_ex_keypoints = np.load(f'src/exercises//{exercise}_delta.npy') 
  files = os.listdir(f'src/screenshots//{exercise}')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join(f"src/screenshots//{exercise}", image) for image in images]

  # Show first jpg
  if current_state == 0:
    image = cv2.imread(correct_ex_jpg[current_state])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Movenet model", image)

  # Check that all keypoints are detected
  coords = np.load(f'src\\exercises\\{exercise}_focus.npy')
  for coord in coords:
    if inputFrame[coord][2] < 0.5:
      print("Not all keypoints are detected.")
      return current_state, duration_states, reps

  scores_current = [0] * 20
  scores_next = [0] * 20

  for coord in coords:
    
    # Determine score of current state
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    np.seterr(divide='ignore')
    scores_current[coord] = np.arctanh(scores_current[coord])

    # Determine score of next state
    scores_next[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + 1) % len(correct_ex_jpg)][coord][:2]))
    np.seterr(divide='ignore')
    scores_next[coord] = np.arctanh(scores_next[coord])

  # Give feedback based on scores
  wrong_position = False
  for i in coords:
    if scores_current[i] <= 2.25:
      wrong_position = True
      print(feedback)

  diff = np.average(scores_current) - np.average(scores_next)
  if (diff < 0) and not wrong_position:
    print("ATTENTION: Movement was not executed right!")
    wrong_position = True
  
  if wrong_position and current_state != 0:
    duration_states += 1

  # If there is more than 10 seconds between 2 states, the exercise fails.
  if duration_states >= 10:
    print("UNSUCCESSFUL: The executing of this exercise was false or took too long!!")
    quit()   
  
  if not wrong_position:
    print("GREAT: This movement was executed perfectly!!")
    # Show image of next state when the execution of the current state was correct
    image = cv2.imread(correct_ex_jpg[(current_state + 1) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Movenet model", image)

    duration_states = 0
    current_state += 1

    if current_state == len(correct_ex_jpg) and reps + 1 == count_reps:
      # show last frame before ending.
      time.sleep(2)

  # Go to next state
  if current_state == len(correct_ex_jpg):
    reps += 1
    current_state = 0

  return current_state, duration_states, reps