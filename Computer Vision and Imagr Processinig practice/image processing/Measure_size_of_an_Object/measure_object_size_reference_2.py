import cv2
from object_detector_1 import *
import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Load Object Detector
detector = HomogeneousBgDetector()

# Load Image
image_path = "C:\PycharmProjects\captured_image3.jpg"

img = cv2.imread(image_path)

# Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Draw polygon around the marker
int_corners = np.intp(corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 3)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# Pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20
cm_per_pixel = 1 / pixel_cm_ratio

# Detect objects
contours = detector.detect_objects(img)

# Apply Gaussian Blur to reduce noise
blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("Gaussian Blur to reduce noise", blurred_img)
cv2.waitKey(0)

# Convert the image to grayscale
gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image to grayscale", gray_img)
cv2.waitKey(0)

# Apply Canny Edge Detection
edges = cv2.Canny(gray_img, 50, 200)
cv2.imshow("Canny Edge Detection", edges)
cv2.waitKey(0)


# Create a blank mask
mask = np.zeros_like(edges)

# Dictionary to store areas
areas = {}

# Draw and calculate areas for each object
for i, cnt in enumerate(contours):
    # Draw contours on the mask
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Calculate the area of the object in pixels
    area_pixels = cv2.countNonZero(mask)

    # Calculate the area in square centimeters
    area_cm2 = area_pixels * (cm_per_pixel ** 2)

    areas[f"Object {i + 1}"] = area_cm2

    # Draw object boundaries
    cv2.polylines(img, [cnt], True, (255, 0, 255), 2)

# Print the areas
for obj, area in areas.items():
    print(f"{obj}: Area = {area:.2f} cm²")

# Convert the mask to a color image for visualization
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Display the images
cv2.imshow("Color Image with Contours", img)
cv2.imshow("Mask with Contours", mask_color)

# Wait and destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


