import pyrealsense2 as rs
import numpy as np
import cv2
import dlib

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 이 부분은 실제 랜드마크 모델 파일 경로로 변경해주세요

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = detector(gray_image)

        # Draw bounding boxes and landmarks for each detected face
        for rect in faces:
            #cv2.rectangle(color_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 3)

            # Get landmarks and draw them on the image
            landmarks = predictor(gray_image, rect)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(color_image, (x, y), 1, (0, 255, 0), -1)

        '''for rect in faces:
            landmarks = predictor(gray_image, rect)

            # Extract eye and mouth coordinates
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

            # Draw bounding boxes and landmarks for eyes
            cv2.polylines(color_image, [np.array(left_eye)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(color_image, [np.array(right_eye)], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw bounding box for mouth
            cv2.polylines(color_image, [np.array(mouth)], isClosed=True, color=(0, 255, 0), thickness=2)'''

        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:
    #stop streaming 
    pipeline.stop()

