from matplotlib.pylab import norm
import tensorflow as tf
import numpy as np
import cv2
from models.helper import _keypoints_and_edges_for_display, to_gif, draw_prediction_on_image, \
    init_crop_region, determine_crop_region, run_inference, crop_and_resize, determine_torso_and_body_range, \
    torso_visible
import os
from exercises.compares import compare


current_state = 0
wrong_states = 0

def predict_movenet_for_video(video_path, exercise):
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
        # Get the model prediction
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        return keypoints_with_scores

    # Load the input video file.
    cap = cv2.VideoCapture(video_path)
    # Initialize the frame count
    frame_count = 0

    output_images = []
    output_keypoints = []
    # waar bijhouden?
    current_state = 0
    
    while cap.isOpened():
        
        ret, frame = cap.read()

        if ret:
            image_height, image_width, _ = frame.shape

            # Initialize only during the first frame
            if frame_count == 0:
                crop_region = init_crop_region(image_height, image_width)

            # Crop and resize according to model input and then return the keypoint with scores
            keypoints_with_scores = run_inference(
                movenet, frame, crop_region,
                crop_size=[input_size, input_size])
            output_keypoints.append(keypoints_with_scores[0][0])

            # Compare frame
            similarity_score = compare(keypoints_with_scores[0][0], current_state)
            if similarity_score > 0.1: # een grenswaarde zoeken
                current_state += 1
                wrong_states = 0
            else:
                wrong_states += 1
            if wrong_states > 10: # een waarde zoeken
                print("exit dit hele programma als fout")
            # manier vinden voor aantal frames te wachten vooralleer het als fout beschouwd wordt

            """ # For GIF Visualization
            output_images.append(draw_prediction_on_image(
                frame.astype(np.int32),
                keypoints_with_scores, crop_region=None,
                close_figure=True, output_image_height=300)) """

            # Crops the image for model 
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            #output = np.stack(output_images, axis=0)

            frame_count += 1

        if ret != True:
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    # will be stored as animation.gif
    #to_gif(output, fps=10)
    
    print("Frame count : ", frame_count)

    return output_keypoints



current_state = 0
wrong_states = 0
duration_states = 0

def predict_movenet_for_webcam(exersice):
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

            # Compare frame
            # TODO: this
            similarity_score_current, similarity_score_next = compare(keypoints_with_scores[0][0], current_state, exersice)
            # als score dichter ligt bij next dan is het de volgende state
            if similarity_score_next > similarity_score_current & similarity_score_next > 0.1: # een grenswaarde zoeken
                current_state += 1
                wrong_states = 0
                duration_states = 0
            elif similarity_score_current > 0.1:
                duration_states += 1
            if wrong_states > 10 & current_state != 0: # een waarde zoeken
                print("exit dit hele programma als fout")
            if duration_states > 10 & current_state != 0:
                print("duurt te lang tussen 2 states (te traag)")

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
  

  