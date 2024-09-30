import cv2
import cv2.aruco as aruco

# הגדרת הדיאגרמה של ה-DICT_4X4_1000
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

# ID של התג שברצונך לצייר
tag_id = 919

# גודל התמונה בפיקסלים (118x118 פיקסלים עבור גודל של 1x1 סנטימטר ב-300 DPI)
size_pixels = 118

# יצירת התמונה של התג בגודל שצוין
img = aruco.drawMarker(aruco_dict, tag_id, size_pixels)

# שמירה על התמונה
# cv2.imwrite('C:\\PycharmProjects\\aruco_tag_919_1x1cm.png', img)

# הצגת התמונה
cv2.imshow('ArUco Tag 919 (1x1 cm)', img)
cv2.waitKey(0)
cv2.destroyAllWindows()