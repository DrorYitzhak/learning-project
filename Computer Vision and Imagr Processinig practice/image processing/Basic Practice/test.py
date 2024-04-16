# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
#
# # טעינת תמונה באמצעות OpenCV
# image = cv2.imread("C:\\PycharmProjects\\Sunflower-on-Blue-bkgd-2.jpg")
#
# # המרת תמונה לתלת-מימד
# data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # פיצוץ התמונה לתכונות נפרדות
# data_flat = data.reshape((-1, 3))
#
# # התאמה של הנתונים למודל
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data_flat)
#
# # הגדרת מודל
# gmm = GaussianMixture(n_components=2, random_state=0)
#
# # הכנסת הנתונים למודל
# gmm.fit(data_scaled)
#
# # הדפסת פרמטרים מוערכים
# print("Means:\n", scaler.inverse_transform(gmm.means_))
# print("\nCovariances:\n", gmm.covariances_)
# print("\nWeights:\n", gmm.weights_)
#
# # קווים להצגת התוצאה
# xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
# zz = np.exp(gmm.score_samples(data_scaled))
# zz = zz.reshape(xx.shape)
#
# # תצוגה
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# # תמונה
# ax1.imshow(data)
# ax1.set_title('Original Image')
#
# # קווים ועמודות גרף
# ax2.contour(xx, yy, zz, levels=5, colors='r', alpha=0.6)
# ax2.bar(range(gmm.n_components), gmm.weights_, align='center', color='blue', alpha=0.7)
# ax2.set_title('GMM Contours and Component Weights')
#
# plt.show()

from datetime import datetime
def was_in_the_past(date_str):
    try:
        # פריקת המחרוזת לרכיבים
        year = int(date_str[:4])
        month = int(date_str[4:5])
        day = int(date_str[6:7])
        hour = int(date_str[8:10])
        minute = int(date_str[10:12])
        second = int(date_str[12:18])

        # התאריך שהמשתמש הזין
        user_date = datetime(year, month, day, hour, minute, second)

        # התאריך הנוכחי
        current_date = datetime.now()

        # בדיקה האם התאריך היה בעבר
        if user_date < current_date:
            return True
        else:
            return False
    except ValueError:
        # אין אפשרות להמיר את הקלט לתאריך
        return False

# קלט מהמשתמש
user_input = input("נא הזן תאריך בפורמט YYYYMMDDHHMMSS: ")

# בדיקה האם התאריך שהוזן היה בעבר
if was_in_the_past(user_input):
    print("התאריך היה בעבר.")
else:
    print("התאריך לא היה בעבר או הקלט שגוי.")
