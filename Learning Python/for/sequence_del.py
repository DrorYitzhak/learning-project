def sequence_del(my_str):
    count = 0
    for str1 in my_str[1:]:
        if str1 != my_str[count]:
            print(my_str[count], end="")
        count += 1
    print(str1)
# def sequence_del(my_str):
#     count = 0
#     for str1 in my_str[0:-1]:
#         count += 1
#         if str1 != my_str[count]:
#             print(str1, end="")
#     print(my_str[count])
#     print(count)
sequence_del(input("enter struing: "))

"""הפונקציה מקבלת מחרוזת ומוחקת את האותיות המופיעות ברצף.
 כלומר, הפונקציה מחזירה מחרוזת בה מופיע תו אחד בלבד מכל רצף תווים זהים במחרוזת הקלט."""