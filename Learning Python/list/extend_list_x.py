def extend_list_x(x, y):
    x = [1, 2, 3]
    y = [4, 5, 6]
    print(x)
    print(y)
    x += y
    print(x)
# extend_list_x("list_x","list_y")
extend_list_x("x", "y")

"""הפונקציה מקבלת שתי רשימות list_y, list_x.
 הפונקציה מרחיבה את list_x (משנה אותה) כך שתכיל בתחילתה גם את list_y, ומחזירה את list_x"""