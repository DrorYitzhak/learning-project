
count_animals = 0
class Beloved_animal1:

    def __init__(self, _neam ="Octavio"):
        global count_animals
        count_animals += 1
        self._One_more_year = 1   # age
        self._neam = "Octavio"

    def birthday(self, amount):
        self._One_more_year +=amount

    def get_age(self):
        print(self._One_more_year)

    def get_neam(self):
        print(self._neam)

    def get_count(self):
    #     Beloved_animal1.count_animals += 1
        print(count_animals)



if __name__ == '__main__':

    Dror = Beloved_animal1()
    # Dror.One_more_year = 1
    # Dror.get_count()
    Dror.birthday(10)
    Dror.get_age()
    Dror._neam = "Octaviop"
    Dror.get_neam()
    print('='*10)

    Dror1 = Beloved_animal1()
    Dror1.birthday(15)
    Dror1.get_age()
    Dror1._neam = "Fish"
    Dror1.get_neam()
    Dror1.get_count()



"""כעת, אחרי שרכשתם כלים נוספים ביחידה, שדרגו את המחלקה שיצרתם.

הסתירו את תכונות שם החיה והגיל שלה (תזכורת: _).
אפשרו לקבוע את שם החיה בזמן יצירת המופע.
במידה ולא נקבע שם לחיה בזמן יצירת המופע, דאגו לאתחל את שמה בערך ברירת המחדל "Octavio".
כתבו מתודה בשם set_name המאפשרת לשנות את השם של החיה.
ממשו מתודה בשם get_name המחזירה את השם של החיה.
צרו משתנה בשם count_animals המיועד לספור כמה חיות נוצרו מן המחלקה. חשבו: באיזה סוג משתנה מדובר, והיכן בקוד יש למקם את פעולת הספירה?
כתבו תוכנית ראשית ובה צרו שני מופעים של חיות (אחד עם שם לבחירתכם, והשני בלי).
הדפיסו את השם של כל אחד מהמופעים - בדקו שעבור אחד מהם הודפס השם שנתתם ועבור השני הודפס ערך ברירת המחדל Octavio.
שנו את השם של אחד מהמופעים והדפיסו את שמו החדש לאחר השינוי באמצעות המתודה get_name.
הדפיסו את המשתנה count_animals ובדקו שאכן מתקבל הערך 2 (כי נוצרו מהמחלקה רק שתי חיות)."""