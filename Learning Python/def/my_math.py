def add(num1, num2):
    print("The result is:", num1 + num2)
def compare( num3=2, num4=0):
    if num3 == num4:
        print("ok")
    else:
        print("not")
def main():
    add(3,1)
    compare()

main()
# if __name__ == "__main__":
#     main()
#
# """אם בסוף הפונקציה נבצע קריאה לmain עם התנאי כמו שרשמתי למטה,
#  נוכל להשתמש בתוכנית הזאת כספריה שאם נקרא לה רגיל ללא as ואז הגדרת שם,
#  זאת אומרת קיראה רגילה ואז פשוט את השם של הפונקציה שבתוך הספריה ורק הפונקציה הספציפית הזאת תרוץ ולא כל התוכנית שבתוך הספריה."""
# # if __name__ == "__main__":
# #     main()