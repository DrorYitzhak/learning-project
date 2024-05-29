import cv2
import os
import time

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Directory to save the images
output_dir = 'C:/Users/drory/learning-project/Deep Learning/Learning PyTorch/images_database'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Capture frames from the webcam
image_count = 0
start_time = time.time()
while image_count < 1000:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check if 10 seconds have passed since the last image was saved
    current_time = time.time()
    if current_time - start_time >= 10:
        image_filename = os.path.join(output_dir, f"image_{image_count}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Image {image_count} saved.")
        start_time = current_time
        image_count += 1

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
