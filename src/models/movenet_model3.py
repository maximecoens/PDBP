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
duration_states = 0
output_images = []

def predict_movenet_for_video(video_path, exercise, delta):
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
    # Initialize frames per second
    fps = 30

    output_keypoints = []
    #current_state = 0
    
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
            if frame_count % (fps * delta) == 0:
                # Save the screenshot
                screenshot_path = os.path.join(f'src/screenshots/{exercise}/screenshot_{frame_count}.jpg')
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
    to_gif(output, exercise, fps=10)
    
    print("Frame count : ", frame_count)

    return output_keypoints

  

output = predict_movenet_for_video("src\/testdata\/bovenhandse_bicep_curl.mp4", "test", 0.5)