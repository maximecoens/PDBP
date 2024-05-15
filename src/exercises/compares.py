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
  else:
    return compare_general(inputFrame, current_state, exercise, reps)
  
keypoints_input = []

def compare_bovenhandsecurl(inputFrame, current_state, reps):
  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  # TODO: time delta werken, tussen current state en next state ook meer dan 10 frames?
  states = np.load('src\exercises//upperhand_bicep_curl.npy') # _delta ? zie general
  delta = 10
  score_current = 0
  score_next = 0

  if current_state + delta >= len(states):
    return 1

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load('src\exercises//upperhand_bicep_curl.npy') 
  files = os.listdir(f'src/screenshots//upperhand_bicep_curl')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join("src/screenshots/upperhand_bicep_curl", image) for image in images]

  ## TODO: Testen voor tonen van verschillende foto's + werkt dit? => testen
  # TODO: checken voor specifieke feedback
  delta = 1
  scores_current = [0] * 20
  scores_next = [0] * 20
  for coord in range(5, 11):

    # TODO: nog aanpassen!! score berekenen voor elk punt apart? => zo specifieke feedback?
    scores_current[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    scores_current[coord] = np.arctanh(scores_current[coord])
    np.seterr(divide='ignore')

    scores_next[coord] = np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]))
    scores_next[coord] = np.arctanh(scores_next[coord])
    np.seterr(divide='ignore')

    # TODO: nu verder onderzoeken voor volgende keer
    # Scores berekenen voor verschillende keypoints in verschillende acties
    # specifieke feedback via dict doen => kijken voor connectie tussen 2 verschillende keypoints

  """
      # TODO: oude versie => dit is indented, moet tab naar achter uit comments
      score_current += np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
      score_next += np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]))
    # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
    np.seterr(divide='ignore')
    score_current = np.arctanh(score_current / 6)
    score_next = np.arctanh(score_next / 6)
  """     

  if score_current < score_next:
    # open images view
    # TODO: fix correct_state += 1 ??
    image = cv2.imread(correct_ex_jpg[(current_state + delta) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Correct exercise", image)
    # cv2.waitKey(0)
  
  if current_state == len(correct_ex_jpg) - 1:
    reps += 1

  return score_current, score_next, reps


## TODO: kijken of het werkt voor een general werking met 17 keypoints! => al zittend niet mogelijk
## mogelijks keypoints weergeven?
## TODO: reps

def compare_general(inputFrame, current_state, exercise, reps):
  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  # TODO: time delta werken, tussen current state en next state ook meer dan 10 frames?
  states = np.load(f'src\exercises//{exercise}_delta.npy') 
  delta = 1
  score_current = 0
  score_next = 0

  if current_state + delta >= len(states):
    return 1

  # paths to correct exersice pictures
  correct_ex_keypoints = np.load(f'src\exercises\/{exercise}.npy') 
  files = os.listdir(f'src/screenshots\/{exercise}')
  images = [image for image in files]
  correct_ex_jpg = [os.path.join(f"src/screenshots\/{exercise}", image) for image in images]

  ## TODO: Testen voor tonen van verschillende foto's + werkt dit?
  delta = 1
  # TODO: werken met default 17 keypoints? of vragen welke
  for coord in range(0,17):
    score_current += np.dot(inputFrame[coord][:2], correct_ex_keypoints[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[current_state][coord][:2]))
    score_next += np.dot(inputFrame[coord][:2], correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[(current_state + delta) % len(correct_ex_jpg)][coord][:2]))
  # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
  np.seterr(divide='ignore')
  score_current = np.arctanh(score_current / 17)
  score_next = np.arctanh(score_next / 17)

  if score_current < score_next:
    # open images view
    # TODO: fix correct_state += 1 ??
    image = cv2.imread(correct_ex_jpg[(current_state + delta) % len(correct_ex_jpg)])
    image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Correct exercise", image)
    # cv2.waitKey(0)
  
  if current_state == len(correct_ex_jpg) - 1:
    reps += 1

  return score_current, score_next, reps