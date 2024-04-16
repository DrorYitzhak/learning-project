x = input("entar a word:")
z = x.replace(' ','')
print(z)
t = len(z)
print(t)
if z[-1:-t-1:-1] == z[0:t+1:1]:
    print("ok")
else:
    print("not")

"""כתבו תוכנית שקולטת מהמשתמש מחרוזת ומדפיסה 'OK' אם היא פלינדרום, אחרת 'NOT'."""


