import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cv2
import os


def compare(inputFrame, current_state, exercise, reps):
  # kijken welke oefening met switch
  if exercise == 'upperhand_bicep_curl':
    return compare_bovenhandsecurl(inputFrame, current_state, reps)
keypoints_input = []

def compare_bovenhandsecurl(inputFrame, current_state, reps):
  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  # TODO: time delta werken, tussen current state en next state ook meer dan 10 frames?
  states = np.load('src\exercises//upperhand_bicep_curl.npy') 
  delta = 10
  if current_state + delta >= len(states):
    return 1

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load('src\exercises//upperhand_bicep_curl.npy') 
  files = os.listdir(f'src/screenshots/upperhand_bicep_curl')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join("src/screenshots/upperhand_bicep_curl", image) for image in images]

  ## TODO: Testen voor tonen van verschillende foto's
  delta = 1
  for coord in range(5, 11):
    score_current2 += np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    score_next2 += np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]))
  # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
  np.seterr(divide='ignore')
  score_current2 = np.arctanh(score_current2 / 6)
  score_next2 = np.arctanh(score_next2 / 6)

  if score_current2 < score_next2:
    # open images view
    # TODO: fix correct_state += 1 ??
    image = cv2.imread(correct_ex_jpg[(current_state + delta) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Correct exercise", image)
    # cv2.waitKey(0)
  
  if current_state == len(correct_ex_jpg) - 1:
    reps += 1

  return score_current2, score_next2, reps