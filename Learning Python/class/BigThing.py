"""
פונקציית isinstance

המחלקות שהכרנו קודם לכן מהוות שם של טיפוס, למשל כל מופע של המחלקה Dog הוא מטיפוס Dog.
הפונקציה isinstance משמשת אותנו כדי לבדוק האם אובייקט מסוים הוא אכן מטיפוס מסוים.
דוגמא:
print (isinstance(2, int))
print (isinstance("text", int))
"""

# -------------------------------------------------------------------------------------------------------------

""" 
פונקציית issubclass

פונקציה שימושית נוספת היא הפונקציה issubclass.
הפונקציה מקבלת שני פרמטרים שהם שמות של מחלקות (ובמקרים שונים גם שמות של טיפוסים מובנים).
הפונקציה מחזירה ערך אמת (True) אם הפרמטר הראשון הוא תת-מחלקה של השני, אחרת היא מחזירה שקר (False).
נדגים בעזרת המחלקות Dog ו-Animal שמימשנו קודם לכן.
דוגמא:
print(issubclass(Dog, Animal))
print(issubclass(Animal, Dog))
print(issubclass(Animal, int))
"""

class BigThing:
    def __init__(self, name):
        self.name = name

    def size(self):
        if isinstance(self.name, int):
            return self.name

        elif isinstance(self.name, str or list or dict):
            return len(self.name)

class BigCat(BigThing):
    def __int__(self, name, weight):
        self.name = name

    def size(self):
        super(BigCat, self).size()
        if self.weight > 15:
            return str("Fat")
        elif self.weight > 20:
            return str("Very Fat")
        else:
            return str("ok")
if __name__ == '__main__':
    my_thing = BigThing(5)
    print(my_thing.size())
    # my_c = BigCat(0)
    # print(my_c.size())