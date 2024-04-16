def numbers_letters_count(my_str):
    for y in my_str:
        if y in ["1", "2", "3", "4", "5", "6", "7", "8", "9","0"]:
            num.append(y)
        else:
            list.append(y)
num = []
list = []

numbers_letters_count(input("start my str: "))
print([len(num), len(list)])

"""הפונקציה מקבלת כפרמטר מחרוזת.
הפונקציה מחזירה רשימה שהאיבר הראשון בה מייצג את מספר הספרות במחרוזת,
 והאיבר השני מייצג את מספר האותיות במחרוזת, כולל רווחים, נקודות, סימני פיסוק וכל מה שאינו ספרות."""