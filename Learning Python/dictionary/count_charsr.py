def count_chars(my_str):
    number_of_characters = {}
    for x in my_str:
        y = my_str.count(x)
        number_of_characters[x] = y
    print(number_of_characters)
count_chars(input("Insert a string:"))
"""הפונקציה מקבלת מחרוזת כפרמטר.
הפונקציה מחזירה מילון, כך שכל איבר בו הוא צמד שמורכב ממפתח: תו מן המחרוזת שהועברה (לא כולל רווחים), ומערך: מספר הפעמים שהתו מופיע במחרוזת."""