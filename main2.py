import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from models import movenet_model2 as mm2
from exercises import bovenhandseCurl


print("INFOINFOINFOINFO")

video_path = "test\BovCurlCorrect\/fr90graden.jpg"
output_keypoints = mm2.predict_movenet_for_video(video_path)

if output_keypoints is not None:    
    print("Converted to Gif Successfully")

# om de 6 frames bijhouden
framerate = 6
output_keypoints_input = []
for frame in range(0, len(output_keypoints), framerate):
   output_keypoints_input.append(output_keypoints[frame])
print(output_keypoints_input)



def fill_list():
   file = open("data\Bovenhandse_curl.txt", "w")
   file.writelines(output_keypoints_input)
   print("Gegevens succesvol opgeslagen in de data directory.")
   file.close()


# TODO: verder verloop
#fill_list()

#with open("data\Bovenhandse_curl.txt", 'r') as f:
#   output_keypoints_correct = eval(f.read())
#print(output_keypoints_correct)

