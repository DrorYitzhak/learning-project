class Animal:
    def __init__(self, name):
        self._name = name
        self._hunger = 0


# class Dog(Animal):
#
#     def __int__(self, name, hunger):
#
#         Animal.__int__(self, name, hunger)
#         self.name = name
#         self.hunger = hunger


    #
    # def get_name(self):
    #     return self.name
    #
    # def is_hungry(self):
    #     if self.hunger == 0:
    #         return True
    #     else:
    #         return False
def main():

    test = Animal("sss", 3)
    # test = Dog("sss", 3)
    # test.get_name()


main()
