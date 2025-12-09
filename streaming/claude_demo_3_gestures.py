import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# Create a gesture recognizer instance
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# For drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open camera
cap = cv2.VideoCapture(0)

print("Showing gesture recognition. Press 'q' to quit")
print("Try gestures: Thumbs Up, Victory, Open Palm, Closed Fist, etc.")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp in milliseconds
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_count * 33  # Approximate 30fps
    
    # Recognize gestures
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)
    
    # Draw hand landmarks
    if recognition_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # Convert to format for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            
            # Draw the hand skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Display recognized gestures
    if recognition_result.gestures:
        y_offset = 30
        for idx, gestures in enumerate(recognition_result.gestures):
            if gestures:
                gesture = gestures[0]  # Top gesture
                handedness = recognition_result.handedness[idx][0].category_name
                
                text = f"{handedness}: {gesture.category_name} ({gesture.score:.2f})"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 40
    
    # Display the frame
    cv2.imshow('Gesture Recognition', frame)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()