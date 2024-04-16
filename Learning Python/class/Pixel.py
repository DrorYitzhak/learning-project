class Pixel:
    """
    _x - x coordinate
    _y - y coordinate
    _red - a value between 0 and 255
    _green - a value between 0 and 255
    _blue - a value between 0 and 255
    """
    def __init__(self, x=0, y=0, red=0, green=0, blue=0):
        self.x_coordinate = x
        self.y_coordinate = y
        self.value_to_red = red
        self.value_to_green = green
        self.value_to_blue = blue

    def set_coords(self, x, y):
        self.x= x
        self.y= y

    def set_grayscale(self):
        Average = (self.value_to_red + self.value_to_green + self.value_to_blue)/3
        # print(Average)
        self.value_to_red = int(Average)
        self.value_to_green = int(Average)
        self.value_to_blue = int(Average)

    def print_pixel_info(self):
        if self.value_to_red == 0 and self.value_to_green == 0 and self.value_to_blue > 50:
            print("x:",self.x_coordinate,",y:",self.y_coordinate,",Color:","(",self.value_to_red,",",self.value_to_green,",",self.value_to_blue,")","blue")

        elif self.value_to_red == 0 and self.value_to_green > 50 and self.value_to_blue == 0:
            print("x:",self.x_coordinate,",y:",self.y_coordinate,",Color:","(",self.value_to_red,",",self.value_to_green,",",self.value_to_blue,")","green")

        elif self.value_to_red > 50 and self.value_to_green == 0 and self.value_to_blue == 0:
            print("x:",self.x_coordinate,",y:",self.y_coordinate,",Color:","(",self.value_to_red,",",self.value_to_green,",",self.value_to_blue,")","red")

        else:
            print("x:", self.x_coordinate, ",y:", self.y_coordinate, ",Color:", "(", self.value_to_red, ",",
              self.value_to_green, ",", self.value_to_blue, ")",)


if __name__ == '__main__':
    my_pixel = Pixel(5, 6, 250)
    my_pixel.print_pixel_info()
    my_pixel.set_grayscale()
    my_pixel.print_pixel_info()


