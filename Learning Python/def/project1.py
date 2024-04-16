def String_check():
    x = input("Write a string:")
    z = x.replace(' ', '')
    print(z)
    t = len(z)
    print(t)
    if z[0:t:1] == z[-1]:
        print("ok")
    else:
        print("not")
String_check()