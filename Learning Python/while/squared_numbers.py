def squared_numbers(start, stop):
        while start ** 2 < (stop + 1) ** 2:
                print(start ** 2)
                start += 1
squared_numbers(start=2, stop=4)

"""הפונקציה מקבלת שני מספרים שלמים, start ו-stop (הניחו שמתקיים: start <= stop).
 הפונקציה מחזירה רשימה בה נמצאים כל ריבועי המספרים בין start ל-stop (כולל)."""