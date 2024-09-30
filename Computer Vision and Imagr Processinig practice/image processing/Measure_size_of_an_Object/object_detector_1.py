import cv2

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0.5)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 115, 15)
        # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    def find_aruco_in_objects(self, corners, ids, contours):
        """
        Finds which contour the single Aruco marker is inside and returns the index of the contour.

        :param corners: List of detected Aruco marker corners (only one Aruco marker expected)
        :param ids: List of detected Aruco marker IDs (only one ID expected)
        :param contours: List of detected object contours
        :return: Index of the contour where the Aruco marker is found, or None if not found
        """
        if not corners or not ids or len(corners) != 1 or len(ids) != 1:
            raise ValueError("Expected exactly one Aruco marker")

        aruco_corners = corners[0][0]  # Get the four corners of the Aruco marker
        aruco_id = ids[0]  # Get the ID of the Aruco marker

        for cnt_index, cnt in enumerate(contours):
            for point in aruco_corners:
                if cv2.pointPolygonTest(cnt, tuple(point), False) >= 0:
                    return cnt_index  # Return the index of the contour where the Aruco marker is found

        return None  # Return None if the Aruco marker is not found in an



