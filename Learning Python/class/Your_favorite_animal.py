class Beloved_animal1:
    def __init__(self):
        self.One_more_year = 1   # age
        # self.neam = "Octavios

    def birthday(self, amount):
        self.One_more_year +=amount

    def get_age(self):
        print(self.One_more_year)

    def get_neam(self):
        print(self.neam)

def test():
    instance = Beloved_animal1()
    instance.birthday(10)
    instance.get_age()
    instance.neam = "Octaviop"
    instance.get_neam()

    instance.birthday(15)
    instance.get_age()
    instance.neam = "Fish"
    instance.get_neam()

test()


"""
כתבו מחלקה (Class) המייצגת את החיה האהובה עליכם (לדוגמה תמנון, Octopus).

הוסיפו מתודת אתחול שתכלול את התכונות הבאות: שם החיה (לדוגמה אוקטביו, Octavio) והגיל שלה.
הוסיפו מתודה בשם birthday שתעלה את גיל החיה ב-1.
הוסיפו מתודה בשם get_age שתחזיר את גיל החיה. ***
כתבו תוכנית ראשית ותיצרו בה שתי חיות (כלומר, מופעים).
הפעילו על מופע חיה אחד את המתודה birthday.
הדפיסו את הגיל של שתי החיות למסך.
שימו לב: בשלב זה כל החיות שתיצרו באמצעות התבנית של המחלקה יקבלו את אותו השם שמופיע במתודת האתחול.
 ביחידה הבאה נלמד כיצד נותנים לכל חיה שם שונה בשלב האתחול.

"""