import cv2

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 115, 15)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 2555, 15)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set area thresholds
        min_area = 50000  # Example minimum area
        max_area = 5000000000  # Example maximum area

        # Filter contours based on area
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:  # Use the area thresholds to filter
                filtered_contours.append(cnt)

        if filtered_contours:  # Check if there are any filtered contours
            cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 2)
        # img_height, img_width = frame.shape[:2]
        # max_width = 1200
        # max_height = 1000
        # # Calculate the scaling factor
        # scale = min(max_width / img_width, max_height / img_height)
        #
        # # Resize the image
        # resized_img = cv2.resize(frame, (int(img_width * scale), int(img_height * scale)))
        #
        # # Display the resized image
        # cv2.imshow("window_name", resized_img)

        # # Instead of waiting indefinitely, return and let the window stay open
        # cv2.waitKey(0)  # Adjust the wait time as needed
        # cv2.destroyAllWindows()

        return filtered_contours  # Return only the filtered contours

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



