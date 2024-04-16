def super_program(my_shopping_list):
    def program_number(numb):
        if int(numb) == 1:
            print(my_shopping_list)

        if int(numb) == 2:
            count = 0
            for lis in my_shopping_list:
                if lis in ",":
                    count += 1
            print(count)

        if int(numb) == 3:
            product = input("Type a product name to see if it is listed:")
            if product in my_shopping_list:
                print("Yes")
            else:
                print("No")

        if int(numb) == 4:
            product = input("Type a product name to know how many times it appears in the list:")
            print(my_shopping_list.count(product))

        if int(numb) == 5:
            product = input("Type the name of the finder you want to delete once from the list:")
            y1 = my_shopping_list.find(product)
            t = (my_shopping_list[y1:])
            y2 = t.find(",")
            print(my_shopping_list[0:(y1 - 1)] + t[y2:])

        if int(numb) == 6:
            product = input("Add a new product name to the list:")
            print(my_shopping_list+","+product)


    program_number(input("enter number:"))
super_program(input("Enter a sample shopping list,(Milk,Cottage,Tomatoes):"))

"""כתבו תוכנית שקולטת מהמשתמש מחרוזת אחת המייצגת רשימת מוצרים לקניות, מופרדת בפסיקים ללא רווחים.
דוגמה למחרוזת קלט: "Milk,Cottage,Tomatoes".

התוכנית מבקשת מהמשתמש להקיש מספר (ספרה) בטווח אחת עד תשע (אין צורך לבדוק תקינות קלט).
בהתאם למספר שנקלט, מבצעת אחת מן הפעולות הבאות, על פי הפירוט הבא:

1) הדפסת רשימת המוצרים
2) הדפסת מספר המוצרים ברשימה
3) הדפסת התשובה לבדיקה "האם המוצר נמצא ברשימה?" (המשתמש יתבקש להקיש שם מוצר)
4) הדפסת התשובה לבדיקה "כמה פעמים מופיע מוצר מסוים?" (המשתמש יתבקש להקיש שם מוצר)
5) מחיקת מוצר מהרשימה (המשתמש יתבקש להקיש שם מוצר, רק מוצר אחד ימחק)
6) הוספת מוצר לרשימה (המשתמש יתבקש להקיש שם מוצר)
7) הדפסת כל המוצרים שאינם חוקיים (מוצר אינו חוקי אם אורכו קטן מ-3 או שהוא כולל תווים שאינם אותיות)
הסרת כל הכפילויות הקיימות ברשימה
יציאה
שימו לב, לאחר ביצוע בחירה של המשתמש, המשתמש יחזור לתפריט הראשי עד אשר יבחר ביציאה (יקיש את המספר 9)."""