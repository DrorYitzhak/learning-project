def are_files_equal(file1, file2):
    file1 = open(r"C:\Users\droryitx\OneDrive - Intel Corporation\Desktop\Files intel\Files PyCharm\File 1 for Exercise 1.txt", "r")
    file2 = open(r"C:\Users\droryitx\OneDrive - Intel Corporation\Desktop\Files intel\Files PyCharm\File 1 for Exercise 2.txt", "r")
    y = file2.read()
    x = file1.read()
    if x in y:
        print(True)
    else:
        print(False)
    if 'not' in y:
        print(True)
    else:
        print(False)
    file2.close()
    file1.close()
are_files_equal(r"C:\Users\droryitx\OneDrive - Intel Corporation\Desktop\Files PyCharm\File 1 for Exercise 1.txt", r"C:\Users\droryitx\OneDrive - Intel Corporation\Desktop\Files PyCharm\File 1 for Exercise 2.txt")


# """הפונקציה מקבלת כפרמטרים נתיבים של שני קבצי טקסט (מחרוזות).
# הפונקציה מחזירה אמת (True) אם הקבצים זהים בתוכנם, אחרת מחזירה שקר (False)."""

# """במהלך פתרון נתקלתי בבעיה שפייתון לא הצליח למצוא את המיקום המדויק של הקובץ,
#  אני הכנסתי רק את שם המיקום מבלי לבוסיף את שם הקובץ ככה - r"C:\Users\droryitx\OneDrive - Intel Corporation\Desktop\Files PyCharm"""
