import tensorflow as tf
import numpy as np
from models import movenet_model
from matplotlib import pyplot as plt

# Load the input image.
image_path = 'test/IMG-20240405-WA0023.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)


# Resize and pad the image to keep the aspect ratio and fit the expected size.
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, 192, 192)

# Run model inference.
keypoints_with_scores = movenet_model.movenet(input_image)
print(keypoints_with_scores)

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(
    display_image, 1280, 1280), dtype=tf.int32)
output_overlay = movenet_model.draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)


plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
_ = plt.axis('off')
plt.show()