import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from wayland_automation.mouse_controller import Mouse

mouse = Mouse()

# Download the face landmarker model first:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw face landmarks on the image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected faces
    for face_landmarks in face_landmarks_list:
        # Convert landmarks to the format needed for drawing
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])
        
        # Draw the face mesh tesselation
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        ## Draw the face contours
        #solutions.drawing_utils.draw_landmarks(
        #    image=annotated_image,
        #    landmark_list=face_landmarks_proto,
        #    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        #    landmark_drawing_spec=None,
        #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        #)
        
        ## Draw the irises
        #solutions.drawing_utils.draw_landmarks(
        #    image=annotated_image,
        #    landmark_list=face_landmarks_proto,
        #    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        #    landmark_drawing_spec=None,
        #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        #)
    
    return annotated_image

# Configure face landmarker for video stream
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open camera
cap = cv2.VideoCapture(0)
timestamp = 0

print("Showing face tracking. Press 'q' to quit")
print("Make sure 'face_landmarker.task' model file is in the same directory")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame and detect faces
        detection_result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1
        
        # Draw face landmarks
        if detection_result.face_landmarks:
            # # list, with the num elements equal to num faces:
            # l1 = detection_result.face_landmarks
            # print("1", type(l1), len(l1) )
            # # list, with the num elements equal to num of points on the face:
            # l2 = detection_result.face_landmarks[0]
            # print("2", type(l2), len(l2) )
            pnt = 0 # just above the mouth
            landmark = detection_result.face_landmarks[0][pnt]
            #print("(x,y)", landmark.x, landmark.y)
            mouse.click(int(2880*landmark.x), int(1920*landmark.y))

            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            # Convert back to BGR for OpenCV display
            frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()