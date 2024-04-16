def arrow(my_char, max_length):
    for i in range(1, int(max_length*2)):
        if i <= int(max_length):
            print(str(my_char)*i)
        else:
            print(str(my_char)*(int(max_length)*2-i))
arrow(input("Insert icon:"), input("Insert size:"))

"""הפונקציה מקבלת שני פרמטרים: הראשון תו בודד, השני הוא גודל מקסימלי.
הפונקציה מחזירה מחרוזת המייצגת מבנה של "חץ" (ראו דוגמה),
 הבנוי מתו הקלט, כאשר מרכז החץ (השורה הארוכה ביותר) הוא באורך הגודל שמועבר כפרמטר."""