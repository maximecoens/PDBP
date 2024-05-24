import cv2
import os

# Save the screenshot
screenshot_path = os.path.join(f'src/screenshots/test/screenshot_1.jpg')
cv2.imwrite(screenshot_path, frame)
print(f'Saved screenshot: {screenshot_path}')