import cv2
import dlib
import numpy as np

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 이 부분은 실제 랜드마크 모델 파일 경로로 변경해주세요

# Start capturing
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = detector(gray_image)

        # Draw bounding boxes and landmarks for each detected face
        for rect in faces:
            landmarks = predictor(gray_image, rect)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Show images
        cv2.imshow('Webcam Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()
