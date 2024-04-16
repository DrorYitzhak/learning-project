class Animal:
    def __init__(self, name, hunger):
        self._name = name
        self._hunger = hunger

    def get_name(self):
        return self._name

    def get_hunger(self):
        return self._hunger

    def is_hungry(self):
        if self._hunger == 0:
            return True
        else:
            return False

    def feed(self):
        self._hunger = self._hunger - 1
        return self._hunger


class Dog(Animal):
    def __init__(self, name, hunger):
        Animal.__init__(self, name, hunger)


class Cat(Animal):
    def __init__(self, name, hunger):
        Animal.__init__(self, name, hunger)


class Skunk(Animal):
    def __init__(self, name, hunger):
        Animal.__init__(self, name, hunger)


class Unicorn(Animal):
    def __init__(self, name, hunger):
        Animal.__init__(self, name, hunger)


class Dragon(Animal):
    def __init__(self, name, hunger):
        Animal.__init__(self, name, hunger)


def main():
    # animal = Animal("lulu", 10)
    # print(animal.is_hungry())
    # print(animal.feed())

    obj_num1 = 10
    dog = Dog("lulu", obj_num1)
    obj_1 = dog.get_name()
    dog.get_name()
    # print(dog.is_hungry())
    # print(dog.feed())

    obj_num2 = 3
    cat = Cat("Zelda", obj_num2)
    obj_2 = cat.get_name()
    # print(cat.is_hungry())

    obj_num3 = 0
    skunk = Skunk("Stinky", obj_num3)
    obj_3 = skunk.get_name()
    # print(skunk.is_hungry())
    # print(skunk.feed())

    obj_num4 = 7
    unicorn = Unicorn("Keith", obj_num4)
    obj_4  = unicorn.get_name()
    # print(unicorn.is_hungry())

    obj_num5 = 1450
    dragon = Dragon("Lizzy", obj_num5)
    obj_5 = dragon.get_name()
    # print(dragon.is_hungry())

    # zoo_lst = []
    # zoo_lst.append(dog)

    zoo_lst = [obj_1, obj_2, obj_3, obj_4, obj_5]
    zoo_obj_num = [obj_num1, obj_num2, obj_num3, obj_num4, obj_num5]

    for animal in zoo_lst:

        animal = Animal(animal, animal.get_hunger())

        while animal.is_hungry():
            animal.feed()

main()


"""
טעויות שהיו לי :
1) כתבתי במקום __init__ את הפקודה __int__ (קשה מאוד לראות את הטעות הזאת) 


"""