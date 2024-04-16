def are_lists_equal(list1, list2):
    a = [1.5, 3, 4, 5, 9, 6, 7, 8]
    b = [1.5, 3, 4, 5, 9, 6, 7, 8]
    a.sort()
    b.sort()
    print(a)
    print(b)
    if  a == b:
        print(True)
    else:
        print(False)
are_lists_equal("list1", "list2")
"""הפונקציה מקבלת שתי רשימות המכילות איברים מהטיפוסים int ו-float בלבד.
הפונקציה מחזירה אמת אם שתי הרשימות מכילות בדיוק את אותם האיברים (גם אם בסדר שונה), אחרת, הפונקציה מחזירה שקר."""
