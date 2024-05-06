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
  states = np.load('src\exercises//upperhand_bicep_curl.npy') 
  
  if current_state + 1 >= len(states):
    return 1

  # calculate cosine similarity score for 6 keypoints (30FPS)
  score_current = 0
  score_next = 0
  for coord in range(5, 11):
    score_current += np.dot(inputFrame[coord][:2], states[current_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(states[current_state][coord][:2]))
    score_next += np.dot(inputFrame[coord][:2], states[current_state + 1][coord][:2]) / (norm(inputFrame[coord][:2])*norm(states[current_state + 1][coord][:2]))
  # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
  np.seterr(divide='ignore')
  score_current = np.arctanh(score_current/6)
  score_next = np.arctanh(score_next / 6)
  # TODO: weight aan meegeven
  return score_current, score_next
