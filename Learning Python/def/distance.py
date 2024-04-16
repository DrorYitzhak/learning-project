def distance(num1, num2, num3):
    absolute_num1_2 = abs(num1 - num2)
    print(absolute_num1_2)
    absolute_num1_3 = abs(num1 - num3)
    print(absolute_num1_3)
    absolute_num2_3 = abs(num2 - num3)
    print(absolute_num2_3)
    if (absolute_num1_2<2 or absolute_num1_3<2)  and  (absolute_num2_3>1 and absolute_num1_2>1) or (absolute_num2_3>1 and absolute_num1_3>1):
        print("True")
    else:
        print("False")
distance(num1 = 4, num2 = 7, num3 = 8 )
"""הפונקציה מקבלת שלושה מספרים שלמים: num1, num2, num3.
הפונקציה מחזירה אמת (True) אם מתקיימים שני התנאים:

אחד מהמספרים num2 או num3 "קרוב" ל-num1.
"קרוב" = מרחק אבסולוטי 1.
אחד מהמספרים num2 או num3 "רחוק" משני המספרים האחרים. "רחוק" = מרחק אבסולוטי 2 ומעלה."""