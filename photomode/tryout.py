import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

image_path = 'thumbs_up.jpg'
model_path = 'gesture_recognizer.task'


# Load the input image from an image file.
mp_image = mp.Image.create_from_file(image_path)


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)
with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.
    # Perform gesture recognition on the provided single image.
    # The gesture recognizer must be created with the image mode.
    gesture_recognition_result = recognizer.recognize(mp_image)
    print(gesture_recognition_result)
    
    
