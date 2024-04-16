my_dictionary = { 'first_name':'Mariah', 'last_name':'Carey', 'birth_date':'27.03.1970', 'hobbies':'Sing, Compose, Act'}
def Personal_dictionary(user_number):
    if int(user_number) == 1:
        print(my_dictionary["last_name"])

    if int(user_number) == 2:
        month = my_dictionary["birth_date"]
        print(month[3:5])

    if int(user_number) == 3:
        print("3")

    if int(user_number) == 4:
        month = my_dictionary["hobbies"]
        print(month[15:18])

    if int(user_number) == 5:
        my_dictionary['hobbies'] = 'Sing, Compose, Act, Cooking'
        print(my_dictionary["hobbies"])

    # if int(user_number) == 6:

    if int(user_number) == 7:
        my_dictionary['age'] = '51'
        print(my_dictionary['age'])

Personal_dictionary(input("tap number:"))


"""כתבו תוכנית שמבצעת את הפעולות הבאות, בהתאם לספרה שהקיש המשתמש:

1) הדפיסו למסך את שם המשפחה של מריה.
2) הדפיסו למסך את החודש בו נולדה מריה.
3) הדפיסו למסך את מספר התחביבים שיש למריה.
4) הדפיסו למסך את התחביב האחרון ברשימת התחביבים של מריה.
5) הוסיפו את התחביב "Cooking" לסוף רשימת התחביבים.
6) הפכו את טיפוס תאריך הלידה לטאפל שכולל 3 מספרים (יום, חודש ושנה - משמאל לימין) והדפיסו אותו.
7) הוסיפו מפתח חדש בשם age אשר כולל את גילה של מריה והציגו אותו."""