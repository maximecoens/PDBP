import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm


def compare(inputFrame, current_state, exercise):
  # kijken welke oefening met switch
  if exercise == 'upperhand_bicep_curl':
    return compare_bovenhandsecurl(inputFrame, current_state)
keypoints_input = []

def compare_bovenhandsecurl(inputFrame, current_state):
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

  # calculate cosine similarity score for 6 keypoints (30FPS)
  score_current = 0
  score_next = 0
  for coord in range(5, 11):
    score_current += np.dot(inputFrame[coord][:2], states[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(states[current_state][coord][:2]))
    score_next += np.dot(inputFrame[coord][:2], states[current_state + delta][coord][:2]) / (norm(inputFrame[coord][:2])*norm(states[current_state + delta][coord][:2]))
  # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
  np.seterr(divide='ignore')
  score_current = np.arctanh(score_current/6)
  score_next = np.arctanh(score_next / 6)
  # TODO: weight aan meegeven
  # TODO: specifiekere feedback meegeven! => rechte hoek tussen armen enzo
  # TODO: reps weergeven en vragen
  return score_current, score_next
