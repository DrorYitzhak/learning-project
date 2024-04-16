def String_check():
    x = input("entar a word:")
    z = x.replace(' ', '')
    t = len(z)
    The_rest = (z[ 0 : t-1 : 1 ])
    Latest = (z[ -1 ])
    u = The_rest.find(Latest)
    if u == 4:
        print("True")
    elif u == -1:
        print("False")
String_check()

