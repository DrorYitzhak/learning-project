import time

import cv2
import numpy as np
from object_detector import HomogeneousBgDetector

class ObjectAndArucoDetector:
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_5X5_1000):
        # Initialize Aruco detector parameters
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)

        # Initialize object detector
        self.detector = HomogeneousBgDetector()
    def detect_aruco(self, img):
        # Detect Aruco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)
        if corners:
            # Compute Aruco marker perimeter and area
            aruco_perimeter = cv2.arcLength(corners[0], True)
            aruco_area_pixels = cv2.contourArea(corners[0])
            pixel_cm_ratio = aruco_perimeter / 4  # Assuming square marker
            cm_per_pixel = 1 / pixel_cm_ratio
            return corners, ids, cm_per_pixel, pixel_cm_ratio
        else:
            return None, None, None
    def process_image(self, img_path):
        # Load image
        img = cv2.imread(img_path)
        # height, width, _ = img.shape

        # Define the scale percentage (for example, scale down by 50%)
        scale_percent = 50  # Percentage to scale down
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dimensions = (width, height)
        # Resize the image
        img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

        # Detect Aruco
        corners, ids, cm_per_pixel, pixel_cm_ratio = self.detect_aruco(img)
        if corners is None:
            raise Exception("No Aruco marker detected!")

        # Detect objects
        contours = self.detector.detect_objects(img)
        cnt_index = self.detector.find_aruco_in_objects(corners, ids, contours)

        areas = {}
        aruco_area_cm2 = 0

        # Process each object
        for i, cnt in enumerate(contours):
            mask = np.zeros_like(img[:, :, 0])
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            area_pixels = cv2.countNonZero(mask)
            area_cm2 = area_pixels * (cm_per_pixel ** 2)
            calibration_factor = 1 / area_cm2

            if cnt_index == i:
                aruco_area_cm2 = area_cm2
                text = f"Aruco: {aruco_area_cm2:.2f} cm^2"
                self.object_width = None
            else:
                rect = cv2.minAreaRect(cnt)
                (self.x, self.y), (w, h), angle = rect
                self.object_width = w / pixel_cm_ratio
                self.object_height = h / pixel_cm_ratio
                box = cv2.boxPoints(rect)
                self.box = box.astype(int)

                areas[f"Object {i + 1}"] = area_cm2
                text = f"Object {i + 1}: {area_cm2:.2f} cm^2"

            self._draw_object_contour_and_text(img, cnt, text, width, height)
        return img, aruco_area_cm2, areas, self.object_width, self.object_height
    def _draw_object_contour_and_text(self, img, cnt, text, img_width, img_height):
        cv2.polylines(img, [cnt], True, (255, 0, 255), 2)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = 0, 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = max(0, min(cx - (text_width // 2), img_width - text_width))
        text_y = max(text_height, min(cy - (text_height // 2), img_height - 2))

        cv2.rectangle(img, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + 2), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)


        if isinstance(self.object_width, float):
            cv2.polylines(img, [self.box], True, (255, 0, 0), 2)
            w_text = "Width {} cm".format(round(self.object_width, 1))
            (text_width, text_height), _ = cv2.getTextSize(w_text, font, font_scale, font_thickness)
            w_text_x = max(0, min(cx - (text_width // 2), img_width - text_width))
            w_text_y = max(text_height, min(cy - (text_height // 2), img_height - 2))
            cv2.rectangle(img, (w_text_x - 2, text_y - text_height - 2 + 100), (w_text_x + text_width + 2, w_text_y + 100), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, w_text, (w_text_x, w_text_y + 100), font, font_scale, (0, 0, 0), font_thickness)

            h_text = "Height {} cm".format(round(self.object_height, 1))
            (text_width, text_height), _ = cv2.getTextSize(h_text, font, font_scale, font_thickness)
            h_text_x = max(0, min(cx - (text_width // 2), img_width - text_width))
            h_text_y = max(text_height, min(cy - (text_height // 2), img_height - 2))
            cv2.rectangle(img, (h_text_x - 2, text_y - text_height - 2 + 200), (h_text_x + text_width + 2, h_text_y + 2 + 200), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, h_text, (h_text_x, h_text_y + 200), font, font_scale, (0, 0, 0), font_thickness)



    def show_image_resized(self, image, window_name="Image", max_width=1200, max_height=1000):
        """
        Displays an image resized to fit within specified maximum dimensions.

        :param image: The image to display.
        :param window_name: The name of the window for displaying the image.
        :param max_width: Maximum width for resizing the image.
        :param max_height: Maximum height for resizing the image.
        """
        img_height, img_width = image.shape[:2]

        # Calculate the scaling factor
        scale = min(max_width / img_width, max_height / img_height)

        # Resize the image
        resized_img = cv2.resize(image, (int(img_width * scale), int(img_height * scale)))

        # Display the resized image
        cv2.imshow(window_name, resized_img)

        # Instead of waiting indefinitely, return and let the window stay open
        cv2.waitKey(0)  # Adjust the wait time as needed
        cv2.destroyAllWindows()
        pass


# Example usage of the new class
detector = ObjectAndArucoDetector()
start = time.time()
# for x in [3]:
#     image = f'{x}x{x}.jpg'
image = 'test_5.jpg'
start = time.time()
img, aruco_area, object_areas, object_width, object_height = detector.process_image(f'C:\\PycharmProjects\\{image}')
detector.show_image_resized(img)

# Print areas
print(image)
print(f"Aruco Area: {aruco_area:.3f} cm^2")
for obj, area in object_areas.items():
    print(f"{obj}: {area:.3f} cm^2")
    print(f"{obj}: width {object_width:.3f} cm")
    print(f"{obj}: height {object_height:.3f} cm")
print("")
stop = time.time()
time = stop - start
print(time)