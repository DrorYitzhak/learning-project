import cv2

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

# Capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save the current frame as an image
        save_path = r'C:\PycharmProjects\captured_image_4.jpg'
        cv2.imwrite(save_path, frame)
        print(f"Image saved to {save_path}")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
