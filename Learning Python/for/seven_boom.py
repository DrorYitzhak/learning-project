def seven_boom(end_number):
    for series in range(int(end_number)+1):

        if series%7 == 0 or "7"in str(series):
            print("BOOM")
        else:
            print(series)
seven_boom(input("enter number:" ))

"""הפונקציה מקבלת מספר שלם, end_number.
הפונקציה מחזירה רשימה בתחום המספרים 0 עד end_number כולל, כאשר מספרים מסוימים מוחלפים במחרוזת 'BOOM', אם הם עונים על אחד מהתנאים הבאים:

המספר הוא כפולה של המספר 7.
המספר מכיל את הספרה 7."""