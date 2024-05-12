import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cv2


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

  # paths to correct exersice pictures
  correct_ex_jpg = ["src\/testdata\/UpCurlCorrect\state1.jpg", "src\/testdata\/UpCurlCorrect\state2.jpg", "src\/testdata\/UpCurlCorrect\state3.jpg", "src\/testdata\/UpCurlCorrect\state4.jpg", "src\/testdata\/UpCurlCorrect\state5.jpg"]
  correct_ex_keypoints = [[[0.4318306 , 0.4865633 , 0.4347133 ],
       [0.41950595, 0.49642164, 0.3861875 ],
       [0.42036933, 0.46913046, 0.52424085],
       [0.41912293, 0.5225864 , 0.48725164],
       [0.42187494, 0.4492176 , 0.5650964 ],
       [0.46647584, 0.5630374 , 0.50052655],
       [0.462151  , 0.40760878, 0.57383347],
       [0.5151753 , 0.6111319 , 0.3368791 ],
       [0.4704888 , 0.28905943, 0.43323138],
       [0.6120714 , 0.68702704, 0.19359113],
       [0.46467805, 0.18979336, 0.21744469],
       [0.60755867, 0.5263097 , 0.5251685 ],
       [0.6012388 , 0.43244854, 0.5739937 ],
       [0.712837  , 0.50354576, 0.480344  ],
       [0.6953114 , 0.42978418, 0.5077223 ],
       [0.80220264, 0.51074123, 0.4178401 ],
       [0.8010237 , 0.45015138, 0.38508683]], 
       [[0.4329273 , 0.48615813, 0.4196937 ],
       [0.42212495, 0.49598   , 0.49519983],
       [0.42210788, 0.46996418, 0.5366313 ],
       [0.42461115, 0.5194296 , 0.5918876 ],
       [0.4260716 , 0.4543647 , 0.6684469 ],
       [0.45865375, 0.55764776, 0.48078272],
       [0.46404666, 0.41183987, 0.7037322 ],
       [0.4675039 , 0.65191936, 0.49610347],
       [0.4683035 , 0.28823963, 0.54555714],
       [0.43436345, 0.771123  , 0.22852516],
       [0.4339972 , 0.16695729, 0.4686572 ],
       [0.6044054 , 0.5341672 , 0.484122  ],
       [0.60041153, 0.43653688, 0.58684087],
       [0.70056355, 0.5046905 , 0.44775447],
       [0.6936099 , 0.43580323, 0.52135897],
       [0.8052119 , 0.5021943 , 0.41774908],
       [0.8068106 , 0.4643855 , 0.35064745]],
       [[0.42262897, 0.4744475 , 0.52449274],
       [0.4196167 , 0.4916318 , 0.58317435],
       [0.4159581 , 0.46141893, 0.45297644],
       [0.42143786, 0.5103267 , 0.53366745],
       [0.42129433, 0.44190753, 0.5953091 ],
       [0.4578241 , 0.5382894 , 0.45883626],
       [0.4599213 , 0.41163206, 0.5059588 ],
       [0.47209284, 0.60630065, 0.38729137],
       [0.45951772, 0.29853767, 0.45049152],
       [0.40994942, 0.66397846, 0.25706667],
       [0.4089445 , 0.28672686, 0.36031717],
       [0.58462477, 0.5027671 , 0.40689838],
       [0.5813626 , 0.43242285, 0.50284094],
       [0.67525077, 0.47412   , 0.22751996],
       [0.6722089 , 0.4481492 , 0.3160617 ],
       [0.7710475 , 0.47669894, 0.1862224 ],
       [0.7679998 , 0.48591027, 0.30709854]],
       [[0.42740697, 0.47874874, 0.56750834],
       [0.4180592 , 0.49156648, 0.5003058 ],
       [0.41830847, 0.46449432, 0.4678179 ],
       [0.41972357, 0.5113439 , 0.5742409 ],
       [0.420421  , 0.44621277, 0.6310611 ],
       [0.4564038 , 0.55347216, 0.525915  ],
       [0.46129474, 0.40535748, 0.56388664],
       [0.45868775, 0.65207446, 0.39051273],
       [0.46540588, 0.2833332 , 0.48970953],
       [0.4093134 , 0.76497865, 0.3841221 ],
       [0.42280507, 0.19719967, 0.4016953 ],
       [0.5916599 , 0.5105817 , 0.43263736],
       [0.589995  , 0.43319017, 0.4892685 ],
       [0.6762952 , 0.47688055, 0.23381002],
       [0.6727272 , 0.4395492 , 0.36608824],
       [0.77199787, 0.47613773, 0.17064495],
       [0.7718465 , 0.47299412, 0.2441586 ]],
       [[0.43554938, 0.48290038, 0.51381195],
       [0.42287338, 0.49737814, 0.5176858 ],
       [0.4228625 , 0.46676466, 0.5251467 ],
       [0.42096516, 0.52057636, 0.5654005 ],
       [0.42305312, 0.44801742, 0.668457  ],
       [0.46428895, 0.56394845, 0.6238663 ],
       [0.4601842 , 0.39570487, 0.59703255],
       [0.5111178 , 0.62153506, 0.354855  ],
       [0.46830815, 0.2921376 , 0.517378  ],
       [0.6064023 , 0.6992398 , 0.27446762],
       [0.46716294, 0.18345669, 0.2153764 ],
       [0.6113105 , 0.52476895, 0.650363  ],
       [0.6073619 , 0.41646922, 0.5619227 ],
       [0.7208623 , 0.50688833, 0.52818036],
       [0.7037167 , 0.42432693, 0.6212413 ],
       [0.8079356 , 0.50011384, 0.4989962 ],
       [0.80791306, 0.4333437 , 0.37223217]]]
  correct_state = 0

  ## TODO: check voor andere keer misschien
  score_current2 = 0
  score_next2 = 0
  delta = 10
  for coord in range(5, 11):
    score_current2 += np.dot(inputFrame[coord][:2], correct_ex_keypoints[correct_state][coord][:2]) / (norm(inputFrame[coord][:2])*norm(correct_ex_keypoints[correct_state][coord][:2]))
    score_next2 += np.dot(inputFrame[coord][:2], states[current_state + delta][coord][:2]) / (norm(inputFrame[coord][:2])*norm(states[current_state + delta][coord][:2]))
  # Ignore divide by zero warnings (when score is 6, body position is exactly equal)
  np.seterr(divide='ignore')
  score_current2 = np.arctanh(score_current2/6)
  score_next2 = np.arctanh(score_next2 / 6)

  if score_current > 2.5:
  # open images view
  # TODO: fix correct_state += 1
  image = cv2.imread(correct_ex_jpg[correct_state])
  image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
  cv2.imshow("Correct exercise", image)
  # cv2.waitKey(0)

  return score_current, score_next