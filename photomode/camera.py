import cv2

# Open the default camera (usually 0 for built-in laptop camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 's' to save a picture, 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Display the frame
    cv2.imshow('Laptop Camera', frame)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Press 's' to save
    if key == ord('s'):
        cv2.imwrite('captured_image.jpg', frame)
        print("Picture saved as 'captured_image.jpg'")
    
    # Press 'q' to quit
    elif key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
