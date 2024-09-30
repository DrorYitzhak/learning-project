import cv2
import cv2.aruco as aruco
image_path = "C:\\PycharmProjects\\123456.jpg"
# קריאת התמונה שבה אתה רוצה לזהות את התג
# img = cv2.imread(image_path)
#
# tag_id = 919
# # הגדרת הדיאגרמה של ה-DICT_4X4_50
# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
#
# # הגדרת פרמטרים לזיהוי תגים
# parameters = aruco.DetectorParameters_create()
#
# # זיהוי התגים בתמונה
# corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
#
# # בדיקת התוצאות
# if ids is not None:
#     for i, marker_id in enumerate(ids.flatten()):
#         if marker_id == tag_id:
#             print(f'Tag with ID {tag_id} detected.')
#             # ניתן גם להציג את התג עם קווים מסביב
#             img = aruco.drawDetectedMarkers(img, corners, ids)
#             cv2.imshow('Detected Tags', img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# else:
#     print('No tags detected.')

import cv2
import cv2.aruco as aruco

# הגדרת הדיאגרמה של ה-DICT_4X4_1000
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# קריאת התמונה שבה אתה רוצה לזהות את התג
img = cv2.imread(image_path)

# הגדרת פרמטרים לזיהוי תגים
parameters = aruco.DetectorParameters_create()

# זיהוי התגים בתמונה
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# בדיקת התוצאות
if ids is not None:
    detected_ids = ids.flatten()  # המרת IDs למערך שטוח
    all_ids_detected = set(detected_ids)  # המרת רשימת ה-IDs לסט לבדיקת נוכחות
    all_ids_set = set(range(1000))  # יצירת סט עם כל ה-IDs האפשריים
    found_ids = all_ids_set.intersection(all_ids_detected)  # מציאת IDs שנמצאו

    if found_ids:
        print(f'The following tags were detected: {sorted(found_ids)}')
    else:
        print('No tags from the list of 1000 were found.')
else:
    print('No tags detected.')