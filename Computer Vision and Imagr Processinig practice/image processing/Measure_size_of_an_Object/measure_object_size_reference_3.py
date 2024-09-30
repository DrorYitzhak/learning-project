from object_detector_1 import *
import numpy as np
import cv2

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

# Load Object Detector
detector = HomogeneousBgDetector()

# Load Image
image_path = "C:\\PycharmProjects\\12345678910.jpg"
img = cv2.imread(image_path)
height, width, _ = img.shape

# Get Aruco marker
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Compute Aruco marker perimeter and area
aruco_perimeter = cv2.arcLength(corners[0], True)
aruco_area_pixels = cv2.contourArea(corners[0])
pixel_cm_ratio = aruco_perimeter / 4
cm_per_pixel = 1 / pixel_cm_ratio

# Detect objects
contours = detector.detect_objects(img)
cnt_index = detector.find_aruco_in_objects(corners, ids, contours)

# Dictionary to store areas
areas = {}
# Draw and calculate areas for each object
for i, cnt in enumerate(contours):

    # Create a blank mask
    mask = np.zeros_like(img[:, :, 0])
    # Draw contours on the mask
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Calculate the area of the object in pixels
    area_pixels = cv2.countNonZero(mask)

    # Calculate the area in square centimeters
    area_cm2 = area_pixels * (cm_per_pixel ** 2)

    # Determine text for each object
    if cnt_index == i:
        aruco = area_cm2
        text = f"Aruco: {aruco:.2f} cm^2"
    else:
        areas[f"Object {i + 1}"] = area_cm2
        text = f"Object {i + 1}: {area_cm2:.2f} cm^2"

    # Draw object boundaries
    cv2.polylines(img, [cnt], True, (255, 0, 255), 2)  # Magenta

    # Compute the center of the object for placing text
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0

    # Set font and size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # Compute text size and position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = cx - (text_width // 2)
    text_y = cy - (text_height // 2)

    # Adjust text position if it is out of bounds
    text_x = max(0, min(text_x, width - text_width))
    text_y = max(text_height, min(text_y, height - 2))

    # Draw background rectangle for text
    background_color = (255, 255, 255)  # White background
    text_color = (0, 0, 0)  # Black text
    cv2.rectangle(img, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + 2), background_color, cv2.FILLED)

    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

# Print the areas
print(f"Aruco: {aruco:.2f} cm^2")  # Print Aruco Marker area
for obj, area in areas.items():
    print(f"{obj}: Area = {area:.2f} cm^2")  # Print other objects' areas

# Convert the mask to a color image for visualization
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Display the images
cv2.imshow("Color Image with Contours", img)
# cv2.imshow("Mask with Contours", mask_color)

# Wait and destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()



