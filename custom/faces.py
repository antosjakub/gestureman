import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from collections import deque

from wayland_automation.mouse_controller import Mouse

mouse = Mouse()
mouse_pos = deque(maxlen=5)

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

            nose = 4
            lch, rch = 50, 280

            chin = 200
            le, re = 468, 473

            landmark = detection_result.face_landmarks[0][nose]
            nose_x, nose_y = landmark.x, landmark.y
            landmark = detection_result.face_landmarks[0][lch]
            lch_x, lch_y = landmark.x, landmark.y
            landmark = detection_result.face_landmarks[0][rch]
            rch_x, rch_y = landmark.x, landmark.y

            landmark = detection_result.face_landmarks[0][chin]
            ch_x, ch_y = landmark.x, landmark.y
            landmark = detection_result.face_landmarks[0][le]
            le_x, le_y = landmark.x, landmark.y
            landmark = detection_result.face_landmarks[0][re]
            re_x, re_y = landmark.x, landmark.y

            w = rch_x - lch_x
            nose_x_offset = nose_x - lch_x
            nose_x_rel = nose_x_offset / w
            # left, middle, right
            # 0.4, 0.5, 0.6
            nose_x_rel_transf = 5*(nose_x_rel-0.4)
            #print(nose_x_rel_transf)

            e_y = (le_y+re_y)*0.5
            h = ch_y - e_y
            nose_y_offset = nose_y - e_y
            nose_y_rel = nose_y_offset / h
            # top, middle, bottom
            # 0.25, 0.3, 0.35
            nose_y_rel_transf = 10*(nose_y_rel-0.25)
            #print(nose_y_rel_transf)

            if nose_y_rel_transf > 0.99:
                y = 1.0
            elif nose_y_rel_transf < 0.01:
                y = 0.0
            else:
                y = nose_y_rel_transf

            if nose_x_rel_transf > 0.99:
                x = 1.0
            elif nose_x_rel_transf < 0.01:
                x = 0.0
            else:
                x = nose_x_rel_transf

            #print("(x,y)", x, y)
            mouse_pos.append((x,y))
            smooth_x = sum(pos[0] for pos in mouse_pos) / len(mouse_pos)
            smooth_y = sum(pos[1] for pos in mouse_pos) / len(mouse_pos)
            #print("1", x, y)
            #print("2", smooth_x, smooth_y)
            mouse.click(int(2880*smooth_x), int(1920*smooth_y))
            #mouse.click(int(2880*x), int(1920*y))
            #mouse.click(int(2880*x), int(1920*0.7))


            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            # Convert back to BGR for OpenCV display
            frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()