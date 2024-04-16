def format_list(my_list):
    t = len(my_list)
    list = my_list[0:t:2]
    print(list)
    list2 = ' '.join(list)
    print(list2)

format_list(["qwert","yuiop","asdfg","hjkla"])

# str = ' '.join(['hello', 'string', 'functions'])
# print(str)

# x = ["qwe","ert","tyu","qaz","wsx"]
# print(x[0:5:2])
"""הפונקציה מקבלת רשימת מחרוזות באורך זוגי.
 הפונקציה מחזירה מחרוזת המכילה את איברי הרשימה במיקומים הזוגיים, מופרדים בפסיק ורווח, ובנוסף גם את האיבר האחרון עם הכיתוב and לפניו."""