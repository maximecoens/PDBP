from matplotlib.pylab import norm
import numpy as np
import cv2
from models.helper import _keypoints_and_edges_for_display, to_gif, draw_prediction_on_image, \
    init_crop_region, determine_crop_region, run_inference, crop_and_resize, determine_torso_and_body_range, \
    torso_visible
from exercises.compares import compare
import os
import tensorflow as tf


current_state = 0
wrong_states = 0
duration_states = 0
output_images = []
tensor_type = tf.float16

def predict_movenet_for_video(video_path, exercise, delta, model):
    
    model_name, interpreter, input_size, tensor_type = initialize_model(model)

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
        input_image = tf.cast(input_image, dtype=tensor_type)
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
    # Initialize frames per second
    fps = 30

    output_keypoints = []
    
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
            
            # Check if it's time to capture a screenshot
            print(frame_count % round(fps * delta))
            if frame_count % round(fps * delta) == 0:
                # Save the screenshot
                if not os.path.isdir(f'src/screenshots/\{exercise}'):
                    os.mkdir(f'src/screenshots/\{exercise}')
                screenshot_path = os.path.join(f'src/screenshots/\{exercise}/screenshot_{frame_count}.jpg')
                cv2.imwrite(screenshot_path, frame)
                print(f'Saved screenshot: {screenshot_path}') 
                
            # For GIF Visualization
            output_images.append(draw_prediction_on_image(
                frame.astype(np.int32),
                keypoints_with_scores, crop_region=None,
                close_figure=True, output_image_height=300))
            
            # Crops the image for model 
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            output = np.stack(output_images, axis=0)

            frame_count += 1

        if ret != True:
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    # will be stored as a gif
    #output = np.stack(output_images, axis=0)
    to_gif(output, exercise, fps)
    
    print("Frame count : ", frame_count)

    return output_keypoints

def predict_movenet_for_webcam(exercise, reps_count, model):

    model_name, interpreter, input_size, tensor_type = initialize_model(model)

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
        input_image = tf.cast(input_image, dtype=tensor_type)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        return keypoints_with_scores

    # Open webcam.
    cap = cv2.VideoCapture(0)
    # Initialize the frame count
    frame_count = 0

    output_keypoints = []
    current_state = 0
    duration_states = 0 
    reps = 0
    
    while (True):
        
        ret, frame = cap.read()

        if ret:
            cv2.imshow('frame', frame)
            image_height, image_width, _ = frame.shape

            # Initialize only during the first frame
            if frame_count == 0:
                crop_region = init_crop_region(image_height, image_width)

            # Crop and resize according to model input and then return the keypoint with scores
            keypoints_with_scores = run_inference(
                movenet, frame, crop_region,
                crop_size=[input_size, input_size])
            output_keypoints.append(keypoints_with_scores[0][0])

            # Compare frame every second (30fps)
            if frame_count % 30 == 0:
                current_state, duration_states, reps = compare(keypoints_with_scores[0][0], current_state, duration_states, reps, reps_count, exercise)
                print("REPS: ", reps)
                print("STATE: ", current_state)

            # Crops the image for model 
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            frame_count += 1
            # creating 'q' as the quit button for the video 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
              break
            if reps == reps_count:
                print("CONGRATULATIONS: EXERCISE PERFORMED SUCCESFULLY!!")
                break

        if ret != True:
            break

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    
    print("Frame count : ", frame_count)
  

def initialize_model(model):
    model_name = "movenet_lightning_f16"
    match model:
        case 1:
            model_name = "movenet_lightning_f16"
            interpreter = tf.lite.Interpreter(model_path="src\models\movenet_lightning_f16.tflite")
            input_size = 192 
            tensor_type = tf.uint8
        case 2:
            model_name = "movenet_thunder_f16"
            interpreter = tf.lite.Interpreter(model_path="src\models\movenet_thunder_f16.tflite")
            input_size = 256
            tensor_type = tf.uint8
        case 3:
            model_name = "movenet_lightning_int8"
            interpreter = tf.lite.Interpreter(model_path="src\models\movenet_lightning_int8.tflite")
            input_size = 192 
            tensor_type = tf.uint8
        case 4:
            model_name = "movenet_thunder_int8"
            interpreter = tf.lite.Interpreter(model_path="src\models\movenet_thunder_int8.tflite")
            input_size = 256
            tensor_type = tf.uint8
        case 5:
            model_name = "lite-model_movenet_singlepose_lightning_3"
            interpreter = tf.lite.Interpreter(model_path="src\models\lite-model_movenet_singlepose_lightning_3.tflite")
            input_size = 192 
            tensor_type = tf.float32
    return model_name, interpreter, input_size, tensor_type
  