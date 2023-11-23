import pyrealsense2 as rs
import numpy as np
import cv2
import dlib

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

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

        # Draw bounding boxes around detected faces
        for rect in faces:
            cv2.rectangle(color_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 3)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

        if cv2.waitKey(1) == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
opencv_face_detection.py