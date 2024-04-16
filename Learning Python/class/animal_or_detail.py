class Animal:
    def __init__(self, name):
        self._name = name
        self._hunger = 0
        self._fun = 0
    def play(self):
        self._fun += 1
    def eat(self):
        if self._hunger > 0:
            self._hunger -= 1
    def go_to_toilet(self):
        self._hunger += 1


class Dog(Animal):
    def __init__(self, name, fur_color):
        self._name = name
        self._hunger = 0
        self._fun = 0
        self._fur_color = fur_color

    def go_for_a_walk(self):
        self._fun += 2
        print("waff:", self._fun)

def main():
    Fluppy = Dog("Flappy", "Brown")
    Fluppy.play()
    Fluppy.eat()
    Fluppy.go_for_a_walk()
main()




