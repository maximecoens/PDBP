from matplotlib.pylab import norm
import numpy as np 
 
"""
 # create a NumPy array 
arr = np.array([1, 2, 4, 4, 6]) 
 
# save the array to a file 
np.save('my_array.npy', arr) 
"""

def compare_bovenhandsecurl(inputFrame, current_state):
  """ return code 0 is True (juiste beweging)
      return code 1 is False (laatste state)
      return code 2 is False (foute beweging)"""
  # states hier bewaren: uit een test code halen in numpy array steken.
  # TODO: unicodeescape => codec cant decode bytes
  states = np.load('src\exercises//upperhand_bicep_curl.npy') 
  """
  if current_state + 1 >= len(states):
    return 1 """

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


loaded_arr = np.load('src/exercises//upperhand_bicep_curl.npy') 
test_arr = np.load('src/exercises//test_bicep.npy')

current_state = 0
duration_states = 0
wrong_states = 0 

juist = np.array([[0.43386027, 0.4560925 , 0.33033517],
       [0.43353328, 0.4444544 , 0.28550774],
       [0.43657106, 0.46160927, 0.3212833 ],
       [0.4406252 , 0.4305482 , 0.36810508],
       [0.43661147, 0.46135327, 0.41632727],
       [0.468656  , 0.41058123, 0.40580767],
       [0.4644969 , 0.46561268, 0.4543549 ],
       [0.4798875 , 0.38850716, 0.2708109 ],
       [0.47527596, 0.4875069 , 0.25447643],
       [0.45492154, 0.33902842, 0.29866254],
       [0.45628998, 0.52179223, 0.15944308],
       [0.55467534, 0.43686733, 0.4648583 ],
       [0.55534256, 0.48041973, 0.54092425],
       [0.6235218 , 0.4329999 , 0.5603208 ],
       [0.62312853, 0.5065433 , 0.37611458],
       [0.70028865, 0.43235156, 0.41443732],
       [0.700781  , 0.50548786, 0.4735988 ]])

min = np.array([[0.4299618 , 0.4675884 , 0.31991124],
       [0.43046755, 0.46449986, 0.32476896],
       [0.43091404, 0.46319205, 0.28914124],
       [0.43758222, 0.457912  , 0.30289915],
       [0.43755406, 0.45035428, 0.27560982],
       [0.47539997, 0.45426416, 0.36919782],
       [0.4789288 , 0.4446262 , 0.3704162 ],
       [0.49223518, 0.4663295 , 0.20373188],
       [0.4921274 , 0.42993513, 0.27486768],
       [0.45265412, 0.45514697, 0.22971481],
       [0.45393646, 0.43580237, 0.2366199 ],
       [0.5567589 , 0.4664373 , 0.37445757],
       [0.55859816, 0.46807534, 0.3551874 ],
       [0.6172521 , 0.46825913, 0.28938842],
       [0.61834365, 0.49492562, 0.25587037],
       [0.69524944, 0.45205012, 0.45767504],
       [0.69838274, 0.4927896 , 0.36762667]])

plus = np.array([[0.42908013, 0.46250474, 0.32329667],
       [0.4291515 , 0.4625679 , 0.36371082],
       [0.42940626, 0.45525968, 0.3025823 ],
       [0.43572688, 0.46659145, 0.3230655 ],
       [0.43681526, 0.4337469 , 0.36315688],
       [0.46709904, 0.4774606 , 0.34797972],
       [0.47076452, 0.42173254, 0.3816895 ],
       [0.46990007, 0.5326404 , 0.2664812 ],
       [0.4689384 , 0.36862022, 0.27848238],
       [0.44379717, 0.5144228 , 0.19721642],
       [0.44762024, 0.38782048, 0.2811893 ],
       [0.54641855, 0.48490193, 0.3336002 ],
       [0.5501167 , 0.46462038, 0.39061207],
       [0.6132225 , 0.49125895, 0.34372824],
       [0.61233664, 0.49544558, 0.27469507],
       [0.6951785 , 0.46990457, 0.42791712],
       [0.70252866, 0.49522164, 0.29513294]])

min90p2 = np.array([[0.44510972, 0.48774052, 0.467261  ],
       [0.43714982, 0.50508714, 0.41064173],
       [0.437047  , 0.46817768, 0.48371485],
       [0.44020563, 0.5363506 , 0.41535133],
       [0.44039005, 0.44511068, 0.42932197],
       [0.47701865, 0.5593632 , 0.5206814 ],
       [0.47757646, 0.39970663, 0.49199167],
       [0.5549727 , 0.5716461 , 0.23686184],
       [0.52047086, 0.36757013, 0.27844816],
       [0.5939836 , 0.51565534, 0.33127412],
       [0.5851341 , 0.38394597, 0.35594162],
       [0.5754561 , 0.5136803 , 0.3595376 ],
       [0.57118726, 0.41506538, 0.43710515],
       [0.624247  , 0.5119506 , 0.28869554],
       [0.6114007 , 0.4162177 , 0.31056836],
       [0.6835195 , 0.4959511 , 0.33803466],
       [0.67393315, 0.4236174 , 0.46917748]])

juist90p2 = np.array([[0.44175693, 0.4771673 , 0.44252086],
       [0.43720168, 0.49099028, 0.50117457],
       [0.4349905 , 0.46430212, 0.4314976 ],
       [0.43517882, 0.516288  , 0.45626825],
       [0.43663856, 0.4380579 , 0.47197133],
       [0.47426736, 0.5490906 , 0.46827066],
       [0.47418058, 0.40592152, 0.44818383],
       [0.55695826, 0.5788068 , 0.24884358],
       [0.5353401 , 0.38637918, 0.34542462],
       [0.59908247, 0.51633173, 0.27567914],
       [0.58854055, 0.39110485, 0.37565905],
       [0.57863975, 0.5100505 , 0.2883698 ],
       [0.5747911 , 0.42462146, 0.4338824 ],
       [0.64419436, 0.48535573, 0.27852938],
       [0.63061535, 0.4167343 , 0.2622489 ],
       [0.7036738 , 0.48939052, 0.30193198],
       [0.6861286 , 0.43588263, 0.3431882 ]])

for i in range(0, 122, 8):
  similarity_score_current, similarity_score_next = compare_bovenhandsecurl(test_arr[i], current_state)
  print(similarity_score_current, similarity_score_next)
  # als score dichter ligt bij next dan is het de volgende state
  if similarity_score_next > similarity_score_current and similarity_score_next > 2.5: # een grenswaarde zoeken
      current_state += 1
      wrong_states = 0
      duration_states = 0
  elif similarity_score_current < 2.5:
      wrong_states += 1
  else:
     duration_states += 1
  if (wrong_states > 10) and (current_state != 0): # een waarde zoeken
      print("exit dit hele programma als fout")
  if duration_states > 10 and current_state != 0:
      print("duurt te lang tussen 2 states (te traag)")
  i += 1
  print(duration_states, current_state, wrong_states)

 
