from matplotlib.pylab import norm
import tensorflow as tf
import numpy as np
import cv2
from webcamhelper import _keypoints_and_edges_for_display, to_gif, draw_prediction_on_image, \
    init_crop_region, determine_crop_region, run_inference, crop_and_resize, determine_torso_and_body_range, \
    torso_visible
import os

states = [[[0.42547175, 0.4861273 , 0.36412877],
       [0.42217952, 0.47484925, 0.35956025],
       [0.42338848, 0.4910192 , 0.29736868],
       [0.42916566, 0.45890477, 0.4835287 ],
       [0.42777932, 0.5178101 , 0.5461199 ],
       [0.46419972, 0.40586567, 0.43371397],
       [0.46145433, 0.5406748 , 0.48706785],
       [0.4764939 , 0.3599822 , 0.30073813],
       [0.47252017, 0.6200142 , 0.3638225 ],
       [0.46817923, 0.3709451 , 0.30767843],
       [0.46035656, 0.60213155, 0.3226062 ],
       [0.55948246, 0.42431352, 0.49069762],
       [0.5600467 , 0.5242334 , 0.52243406],
       [0.62659824, 0.4276847 , 0.5457316 ],
       [0.62169105, 0.5281274 , 0.39161295],
       [0.6992366 , 0.43517724, 0.39006683],
       [0.7020416 , 0.5385123 , 0.49296486]],
    [[0.43709433, 0.49212703, 0.3909963 ],
       [0.43275294, 0.4841948 , 0.4141657 ],
       [0.4351961 , 0.49533004, 0.37195754],
       [0.4383676 , 0.46042052, 0.47347787],
       [0.43758076, 0.5249137 , 0.51598096],
       [0.46104085, 0.40785033, 0.44931942],
       [0.46128005, 0.53893787, 0.539232  ],
       [0.45678473, 0.3240213 , 0.3114101 ],
       [0.44969225, 0.63465416, 0.29731473],
       [0.43448758, 0.20125414, 0.35606283],
       [0.44680497, 0.63530475, 0.40409836],
       [0.55174524, 0.41372925, 0.55704916],
       [0.55358297, 0.5228789 , 0.6537844 ],
       [0.6242438 , 0.4214813 , 0.5306207 ],
       [0.62578404, 0.5195143 , 0.39666843],
       [0.7083765 , 0.42886487, 0.39222178],
       [0.7072756 , 0.5179113 , 0.5499265 ]],
    [[0.4219889 , 0.460606  , 0.27541554],
       [0.42297852, 0.44754705, 0.32806283],
       [0.42082006, 0.4782856 , 0.20096187],
       [0.4249072 , 0.4309595 , 0.46709153],
       [0.42136806, 0.49399364, 0.40027052],
       [0.45264253, 0.39951354, 0.4153561 ],
       [0.44611   , 0.5264446 , 0.42058596],
       [0.4563356 , 0.30964437, 0.30392385],
       [0.4593382 , 0.57958263, 0.3812432 ],
       [0.4413583 , 0.2613711 , 0.33883855],
       [0.4545082 , 0.60318595, 0.35776162],
       [0.53641915, 0.41696072, 0.5348345 ],
       [0.53565025, 0.5103782 , 0.5444485 ],
       [0.60892737, 0.42569026, 0.5027416 ],
       [0.6133872 , 0.5245413 , 0.45551035],
       [0.70107675, 0.42610377, 0.45356908],
       [0.69802415, 0.5214258 , 0.5246661 ]],
    [[0.4348503 , 0.45771825, 0.29734004],
       [0.43430275, 0.4367221 , 0.27037883],
       [0.43630916, 0.4698945 , 0.2848319 ],
       [0.44109696, 0.41370544, 0.35117486],
       [0.43801826, 0.4843976 , 0.46452543],
       [0.47282428, 0.38278192, 0.47529483],
       [0.46611005, 0.4951701 , 0.44180882],
       [0.48224813, 0.352661  , 0.25072998],
       [0.47634265, 0.5367599 , 0.32450795],
       [0.45750946, 0.32534695, 0.308202  ],
       [0.45642662, 0.5524577 , 0.2151274 ],
       [0.5577384 , 0.4088051 , 0.37001893],
       [0.5587255 , 0.49529022, 0.35596848],
       [0.6257271 , 0.41939592, 0.5338739 ],
       [0.6250758 , 0.5185258 , 0.434238  ],
       [0.6991195 , 0.4233379 , 0.43714467],
       [0.70157576, 0.5160031 , 0.4926937 ]]]

def compare2(correctFrame, inputFrame):

  # calculate cosine similarity score for 6 keypoints (30FPS)
    
  result = 0
  for coord in range(5, 11):
    result += np.dot(correctFrame[coord][:2], inputFrame[coord][:2]) / (norm(correctFrame[coord][:2])*norm(inputFrame[coord][:2]))

  result /= 6

  return result


current_state = 0
wrong_states = 0

def predict_movenet_for_webcam():
    model_name = "movenet_lightning"
    interpreter = tf.lite.Interpreter(model_path="src\models\lite-model_movenet_singlepose_lightning_3.tflite")
    input_size = 192 

    interpreter.allocate_tensors()

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        return keypoints_with_scores

    # Load the input video file.
    cap = cv2.VideoCapture(0)
    # Initialize the frame count
    frame_count = 0

    output_images = []
    output_keypoints = []
    
    while (True):
        
        ret, frame = cap.read()

        if ret:
            cv2.imshow('frame', frame)
            # creating 'q' as the quit button for the video 
            image_height, image_width, _ = frame.shape

            # Initialize only during the first frame
            if frame_count == 0:
                crop_region = init_crop_region(image_height, image_width)

            # Crop and resize according to model input and then return the keypoint with scores
            keypoints_with_scores = run_inference(
                movenet, frame, crop_region,
                crop_size=[input_size, input_size])
            output_keypoints.append(keypoints_with_scores[0][0])

            # hier komt code voor comparison en state bijhouden
            cossim = compare2(keypoints_with_scores[0][0], states[0])
            print(cossim)

            # For GIF Visualization TODO: bekijk voor real-time visualization
            #output_images.append(draw_prediction_on_image(
            #    frame.astype(np.int32),
            #    keypoints_with_scores, crop_region=None,
            #    close_figure=True, output_image_height=300))

            # Crops the image for model 
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            #output = np.stack(output_images, axis=0)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): 
              break

        if ret != True:
            break

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    
    # will be stored as animation.gif
    #to_gif(output, fps=10)
    
    print("Frame count : ", frame_count)

    return output_keypoints


keypoints = predict_movenet_for_webcam()
print(keypoints)
  
