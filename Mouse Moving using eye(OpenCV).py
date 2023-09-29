import cv2                                  # Import OpenCV
import mediapipe as mp                      # Import MediaPipe
import pyautogui                            # Import PyAutoGUI

capture = cv2.VideoCapture(0)               # Initialize the camera capture
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)  # Initialize FaceMesh for facial landmark detection
screen_width, screen_height = pyautogui.size()  # Get screen size for cursor movement

while True:
    _, frame = capture.read()               # Capture a frame from the camera
    frame = cv2.flip(frame, 1)             # Flip the frame horizontally for an intuitive view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for FaceMesh processing
    output = face_mesh.process(rgb_frame)  # Process the frame with FaceMesh for landmark detection
    landmark_points = output.multi_face_landmarks  # Get detected landmark points
    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    key = cv2.waitKey(1) & 0xFF             # Check if the 'q' key is pressed to exit the loop
    if key == ord('q'):
        break

    if landmark_points:
        landmarks = landmark_points[0].landmark  # Extract facial landmarks

        # Draw green circles around specific landmarks (474 to 477)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))  # Draw green circles
            if id == 1:
                screen_x = screen_width * landmark.x
                screen_y = screen_height * landmark.y
                pyautogui.moveTo(screen_x, screen_y)  # Move cursor to corresponding position

        left_eye_landmarks = [landmarks[145], landmarks[159]]  # Extract left eye landmarks (145 and 159)

        # Draw yellow circles around left eye landmarks
        for landmark in left_eye_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))  # Draw yellow circles

        # Check if the vertical difference between two left eye landmarks is small
        if (left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.004:
            pyautogui.click()               # Simulate a left mouse click
            pyautogui.sleep(1)             # Sleep for 1 second to prevent multiple clicks in quick succession

    cv2.imshow('Eye Controlled Mouse', frame)  # Display the frame with annotations

capture.release()                          # Release the camera
cv2.destroyAllWindows()                    # Close the OpenCV window
