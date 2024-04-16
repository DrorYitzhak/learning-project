def is_greater(my_list, n):
    for list in my_list:
        if list>n:
            print(list)

is_greater([1,2,3,4,5,6,7],3)
"""הפונקציה מקבלת שני פרמטרים: רשימה ומספר n.
הפונקציה מחזירה רשימה חדשה ובה כל האיברים שגדולים מהמספר n."""