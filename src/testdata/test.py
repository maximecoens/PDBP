import re
import numpy as np

lijst = ["hallo_niets", "halloe_weliets"]
for i, ex in enumerate(lijst):
  ex = re.sub("_", " ", ex.capitalize())
  print(ex) 

exercises = np.load(f'src\exercises\/exercises.npy')
print(exercises)